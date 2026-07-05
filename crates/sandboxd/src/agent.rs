//! Agent-run orchestration: create a sandbox, run a coding-agent CLI, collect a
//! diff, and optionally turn that diff into a GitHub pull request.

use crate::auth::AuthContext;
use crate::error::{ApiError, ApiResult};
use crate::ids;
use crate::model::{
    AgentGithubConfig, AgentKind, AgentRun, AgentRunState, CreateAgentRunRequest,
    CreateSandboxRequest, Hardness,
};
use crate::runtime::ExecRequest;
use crate::secrets;
use crate::service;
use crate::state::AppState;
use crate::templates::{create_request_from_value, merge_create};
use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

const MAX_LOG_BYTES: usize = 1024 * 1024;
const MAX_DIFF_BYTES: usize = 4 * 1024 * 1024;

struct HardnessProfile {
    create: Value,
    timeout_seconds: u64,
    loop_iterations: u32,
}

pub fn start_agent_run(
    state: AppState,
    ctx: AuthContext,
    req: CreateAgentRunRequest,
) -> ApiResult<AgentRun> {
    validate_secret_exists(&state, &ctx.org_id, &req.api_key_secret)?;
    if let Some(github) = &req.github {
        if let Some(token_secret) = &github.token_secret {
            validate_secret_exists(&state, &ctx.org_id, token_secret)?;
        }
    }
    if req.repo.url.trim().is_empty() {
        return Err(ApiError::BadRequest("repo.url cannot be empty".into()));
    }
    if req.prompt.trim().is_empty() {
        return Err(ApiError::BadRequest("prompt cannot be empty".into()));
    }
    if req.model.trim().is_empty() {
        return Err(ApiError::BadRequest("model cannot be empty".into()));
    }
    if !secrets::valid_name(&req.api_key_secret) {
        return Err(ApiError::BadRequest(
            "api_key_secret must be an env-style secret name".into(),
        ));
    }

    let now = Utc::now();
    let run = AgentRun {
        id: ids::agent_run_id(),
        org_id: ctx.org_id.clone(),
        state: AgentRunState::Queued,
        sandbox_id: None,
        template: req.template,
        repo: req.repo,
        prompt: req.prompt,
        model: req.model,
        agent: req.agent,
        api_key_secret: req.api_key_secret,
        hardness: req.hardness.unwrap_or_default(),
        r#loop: req.r#loop.unwrap_or_default(),
        github: req.github,
        stdout: String::new(),
        stderr: String::new(),
        diff: String::new(),
        logs_truncated: false,
        verification_result: None,
        branch: None,
        commit: None,
        pr_url: None,
        error: None,
        created_at: now,
        updated_at: now,
        finished_at: None,
    };
    state
        .store
        .put_agent_run(&run)
        .map_err(ApiError::Internal)?;

    let run_id = run.id.clone();
    tokio::spawn(async move {
        if let Err(e) = run_agent_task(state.clone(), ctx, run_id.clone()).await {
            tracing::error!(agent_run = %run_id, error = ?e, "agent run failed");
            if let Ok(Some(mut run)) = state.store.get_agent_run(&run_id) {
                let now = Utc::now();
                run.state = AgentRunState::Failed;
                run.error = Some(e.to_string());
                run.updated_at = now;
                run.finished_at = Some(now);
                let _ = state.store.put_agent_run(&run);
            }
        }
    });

    Ok(run)
}

async fn run_agent_task(state: AppState, ctx: AuthContext, run_id: String) -> Result<()> {
    let mut run = state
        .store
        .get_agent_run(&run_id)?
        .ok_or_else(|| anyhow!("agent run {run_id} not found"))?;
    run.state = AgentRunState::Running;
    run.updated_at = Utc::now();
    state.store.put_agent_run(&run)?;

    let create_value = build_create_value(&state, &run)?;
    let create_req: CreateSandboxRequest = create_request_from_value(create_value)?;
    let sb = service::create_sandbox(&state, &ctx, create_req).await?;
    run.sandbox_id = Some(sb.id.clone());
    run.updated_at = Utc::now();
    state.store.put_agent_run(&run)?;

    let handle = sb
        .runtime_handle
        .clone()
        .ok_or_else(|| anyhow!("created sandbox has no runtime handle"))?;
    let node = state.node_for(sb.node_id.as_deref().unwrap_or(""));
    let api_secret_value = decrypt_secret(&state, &run.org_id, &run.api_key_secret)?;

    node.write_file(
        &handle,
        ".workdir-agent-prompt.txt",
        agent_prompt(&run).as_bytes(),
    )
    .await
    .context("write agent prompt")?;

    let profile = hardness_profile(run.hardness);
    let iterations = effective_iterations(&run, profile.loop_iterations);
    let mut last_exit = 0;
    let mut stdout = String::new();
    let mut stderr = String::new();
    for _ in 0..iterations {
        let cmd = agent_command(&run);
        let env = agent_env(run.agent, &api_secret_value);
        let exec = tokio::time::timeout(
            Duration::from_secs(profile.timeout_seconds),
            node.exec(
                &handle,
                &ExecRequest {
                    cmd,
                    cwd: None,
                    env,
                    background: false,
                },
            ),
        )
        .await
        .map_err(|_| anyhow!("agent timed out after {}s", profile.timeout_seconds))?
        .context("run agent command")?;
        last_exit = exec.exit_code;
        stdout.push_str(&exec.stdout);
        stderr.push_str(&exec.stderr);
        if exec.exit_code == 0 {
            break;
        }
    }

    let stdout = redact_secret(stdout, Some(&api_secret_value));
    let stderr = redact_secret(stderr, Some(&api_secret_value));
    let (stdout, stdout_truncated) = truncate(stdout, MAX_LOG_BYTES);
    let (stderr, stderr_truncated) = truncate(stderr, MAX_LOG_BYTES);
    run.stdout = stdout;
    run.stderr = stderr;
    run.logs_truncated = stdout_truncated || stderr_truncated;
    run.updated_at = Utc::now();
    state.store.put_agent_run(&run)?;

    if last_exit != 0 {
        bail!("agent exited with code {last_exit}");
    }

    let cleanup = node
        .exec(
            &handle,
            &ExecRequest {
                cmd: "rm -f .workdir-agent-prompt.txt .workdir-agent-final.txt; rm -rf .workdir-codex-home".to_string(),
                cwd: None,
                env: BTreeMap::new(),
                background: false,
            },
        )
        .await
        .context("clean agent internal files")?;
    if cleanup.exit_code != 0 {
        bail!("clean agent internal files failed: {}", cleanup.stderr);
    }

    let diff = node
        .exec(
            &handle,
            &ExecRequest {
                cmd: "git add -A && git diff --cached --binary".to_string(),
                cwd: None,
                env: BTreeMap::new(),
                background: false,
            },
        )
        .await
        .context("collect git diff")?;
    if diff.exit_code != 0 {
        bail!("collect git diff failed: {}", diff.stderr);
    }
    if diff.stdout.trim().is_empty() {
        bail!("agent produced no git diff");
    }
    let diff_stdout = redact_secret(diff.stdout, Some(&api_secret_value));
    let (stored_diff, diff_truncated) = truncate(diff_stdout, MAX_DIFF_BYTES);
    run.diff = stored_diff.clone();
    run.logs_truncated = run.logs_truncated || diff_truncated;
    run.verification_result = Some("not_configured".into());
    run.updated_at = Utc::now();
    state.store.put_agent_run(&run)?;

    if let Some(github) = run.github.clone().filter(|g| g.token_secret.is_some()) {
        let token_secret = github.token_secret.as_deref().unwrap();
        let token = decrypt_secret(&state, &run.org_id, token_secret)?;
        let pr = create_github_pr(&state, &run, &github, &stored_diff, &token).await?;
        run.branch = Some(pr.branch);
        run.commit = Some(pr.commit);
        run.pr_url = Some(pr.url);
    }

    let now = Utc::now();
    run.state = AgentRunState::Succeeded;
    run.updated_at = now;
    run.finished_at = Some(now);
    state.store.put_agent_run(&run)?;
    Ok(())
}

fn build_create_value(state: &AppState, run: &AgentRun) -> ApiResult<Value> {
    let base = if let Some(template) = &run.template {
        state
            .store
            .get_template_by_name(&run.org_id, template)
            .map_err(ApiError::Internal)?
            .ok_or_else(|| ApiError::NotFound(format!("template {template}")))?
            .create
    } else {
        hardness_profile(run.hardness).create
    };
    let mut create = merge_create(base, None)?;
    set_repo_git(&mut create, run);
    Ok(create)
}

fn set_repo_git(create: &mut Value, run: &AgentRun) {
    let startup = startup_object(create);
    let mut git = json!({
        "url": run.repo.url,
        "depth": 1,
    });
    if let Some(r) = &run.repo.r#ref {
        git["ref"] = Value::String(r.clone());
    }
    startup.insert("git".into(), git);
}

fn startup_object(create: &mut Value) -> &mut Map<String, Value> {
    if !create.is_object() {
        *create = json!({});
    }
    let root = create.as_object_mut().expect("create is object");
    let replace = !root.get("startup").map(|v| v.is_object()).unwrap_or(false);
    if replace {
        root.insert("startup".into(), json!({}));
    }
    root.get_mut("startup")
        .and_then(Value::as_object_mut)
        .expect("startup is object")
}

fn hardness_profile(h: Hardness) -> HardnessProfile {
    match h {
        Hardness::Easy => HardnessProfile {
            create: json!({
                "image": "node-python",
                "resources": { "cpu": 1, "memory_mb": 2048, "disk_gb": 16 },
                "auto_stop_seconds": 600
            }),
            timeout_seconds: 900,
            loop_iterations: 1,
        },
        Hardness::Medium => HardnessProfile {
            create: json!({
                "image": "node-python",
                "resources": { "cpu": 2, "memory_mb": 4096, "disk_gb": 16 },
                "auto_stop_seconds": 1200
            }),
            timeout_seconds: 1800,
            loop_iterations: 2,
        },
        Hardness::Hard => HardnessProfile {
            create: json!({
                "image": "heavy-build",
                "resources": { "cpu": 4, "memory_mb": 8192, "disk_gb": 32 },
                "auto_stop_seconds": 3600
            }),
            timeout_seconds: 3600,
            loop_iterations: 3,
        },
    }
}

fn effective_iterations(run: &AgentRun, profile_default: u32) -> u32 {
    match run.r#loop.max_iterations {
        Some(v) => v.clamp(1, 5),
        None if run.r#loop.goal.is_some() => profile_default.clamp(1, 5),
        None => 1,
    }
}

fn agent_prompt(run: &AgentRun) -> String {
    let mut prompt = String::new();
    if let Some(goal) = &run.r#loop.goal {
        prompt.push_str("Goal:\n");
        prompt.push_str(goal);
        prompt.push_str("\n\n");
    }
    prompt.push_str(&run.prompt);
    prompt
}

fn agent_env(agent: AgentKind, api_key: &str) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    match agent {
        AgentKind::Codex => {
            env.insert("CODEX_API_KEY".into(), api_key.to_string());
            // Older docs and Workdir examples used OPENAI_API_KEY. Keep it
            // available for compatibility with existing templates and tests.
            env.insert("OPENAI_API_KEY".into(), api_key.to_string());
        }
        AgentKind::ClaudeCode => {
            env.insert("ANTHROPIC_API_KEY".into(), api_key.to_string());
        }
    }
    env
}

fn agent_command(run: &AgentRun) -> String {
    let model = shell_quote(&run.model);
    match run.agent {
        AgentKind::Codex => {
            let codex_home = shell_quote(&format!("/tmp/workdir-codex-home-{}", run.id));
            format!(
            "set -e; export PATH=\"$PWD:$PATH\"; \
             export CODEX_HOME={codex_home}; mkdir -p \"$CODEX_HOME\"; \
             if ! command -v codex >/dev/null 2>&1; then npm install -g @openai/codex; fi; \
             codex exec --model {model} --ignore-user-config \
               --dangerously-bypass-approvals-and-sandbox \
               --cd \"$PWD\" --skip-git-repo-check --output-last-message /tmp/workdir-agent-final.txt - \
               < .workdir-agent-prompt.txt"
            )
        }
        AgentKind::ClaudeCode => format!(
            "set -e; export PATH=\"$PWD:$PATH\"; \
             if ! command -v claude >/dev/null 2>&1; then npm install -g @anthropic-ai/claude-code; fi; \
             claude --print --model {model} --permission-mode bypassPermissions \
               --dangerously-skip-permissions < .workdir-agent-prompt.txt \
               | tee /tmp/workdir-agent-final.txt"
        ),
    }
}

fn validate_secret_exists(state: &AppState, org_id: &str, name: &str) -> ApiResult<()> {
    if !secrets::valid_name(name) {
        return Err(ApiError::BadRequest(format!(
            "secret '{name}' must be an env-style identifier"
        )));
    }
    state
        .store
        .get_secret(org_id, name)
        .map_err(ApiError::Internal)?
        .ok_or_else(|| ApiError::BadRequest(format!("secret '{name}' is not defined")))?;
    Ok(())
}

fn decrypt_secret(state: &AppState, org_id: &str, name: &str) -> Result<String> {
    let rec = state
        .store
        .get_secret(org_id, name)?
        .ok_or_else(|| anyhow!("secret '{name}' is not defined"))?;
    crate::secrets::decrypt(&state.secret_key, &rec)
}

fn truncate(mut value: String, max: usize) -> (String, bool) {
    if value.len() <= max {
        return (value, false);
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    (value, true)
}

fn redact_secret(value: String, secret: Option<&str>) -> String {
    match secret.filter(|s| !s.is_empty()) {
        Some(secret) => value.replace(secret, "[redacted]"),
        None => value,
    }
}

struct PullRequestResult {
    branch: String,
    commit: String,
    url: String,
}

async fn create_github_pr(
    state: &AppState,
    run: &AgentRun,
    github: &AgentGithubConfig,
    diff: &str,
    token: &str,
) -> Result<PullRequestResult> {
    let repo = GithubRepo::parse(&run.repo.url)?;
    let base = github
        .base_branch
        .clone()
        .or_else(|| run.repo.r#ref.clone())
        .unwrap_or_else(|| "main".to_string());
    let branch = format!("workdir/{}", run.id.replace('_', "-"));
    let root = std::env::temp_dir().join(format!("workdir-agent-{}", run.id));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).context("create agent temp dir")?;
    let repo_dir = root.join("repo");
    let patch_path = root.join("agent.patch");
    std::fs::write(&patch_path, diff).context("write agent patch")?;
    let askpass = root.join("askpass.sh");
    std::fs::write(
        &askpass,
        "#!/bin/sh\ncase \"$1\" in *Username*) echo x-access-token ;; *) echo \"$GITHUB_TOKEN\" ;; esac\n",
    )
    .context("write git askpass")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&askpass, std::fs::Permissions::from_mode(0o700)).ok();
    }

    let clone_url = format!("https://github.com/{}/{}.git", repo.owner, repo.name);
    run_host_cmd(
        &root,
        &askpass,
        token,
        "git",
        &[
            "clone", "--depth", "1", "--branch", &base, &clone_url, "repo",
        ],
    )
    .await
    .context("clone repo")?;
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["checkout", "-b", &branch],
    )
    .await
    .context("create branch")?;
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["apply", "--index", "--binary", patch_path.to_str().unwrap()],
    )
    .await
    .context("apply agent patch")?;
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["config", "user.name", "workdir-agent"],
    )
    .await?;
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["config", "user.email", "agent@workdir.dev"],
    )
    .await?;
    let message = format!("workdir agent run {}", run.id);
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["commit", "-m", &message],
    )
    .await
    .context("commit agent patch")?;
    let commit = run_host_cmd(&repo_dir, &askpass, token, "git", &["rev-parse", "HEAD"])
        .await
        .context("read commit")?
        .trim()
        .to_string();
    run_host_cmd(
        &repo_dir,
        &askpass,
        token,
        "git",
        &["push", "origin", &format!("HEAD:refs/heads/{branch}")],
    )
    .await
    .context("push branch")?;

    let title = format!("Workdir agent run {}", run.id);
    let body = format!(
        "Created by workdir agent run `{}`.\n\nAgent: `{}`\nModel: `{}`\nHardness: `{}`",
        run.id,
        run.agent.as_str(),
        run.model,
        run.hardness.as_str()
    );
    let res = state
        .http
        .post(format!(
            "https://api.github.com/repos/{}/{}/pulls",
            repo.owner, repo.name
        ))
        .bearer_auth(token)
        .header("User-Agent", "workdir-agent")
        .json(&json!({
            "title": title,
            "head": branch,
            "base": base,
            "body": body,
            "draft": github.draft.unwrap_or(true),
        }))
        .send()
        .await
        .context("create GitHub PR")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        bail!("create GitHub PR failed ({status}): {body}");
    }
    let pr: Value = res.json().await.context("parse GitHub PR response")?;
    let url = pr
        .get("html_url")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("GitHub PR response did not include html_url"))?
        .to_string();
    let _ = std::fs::remove_dir_all(&root);
    Ok(PullRequestResult {
        branch,
        commit,
        url,
    })
}

async fn run_host_cmd(
    cwd: &Path,
    askpass: &Path,
    token: &str,
    program: &str,
    args: &[&str],
) -> Result<String> {
    let out = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", askpass)
        .env("GITHUB_TOKEN", token)
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "{} {} failed: {}",
            program,
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

struct GithubRepo {
    owner: String,
    name: String,
}

impl GithubRepo {
    fn parse(url: &str) -> Result<GithubRepo> {
        let mut s = url.trim().trim_end_matches('/').to_string();
        if let Some(rest) = s.strip_prefix("git@github.com:") {
            s = format!("https://github.com/{rest}");
        }
        let marker = "github.com/";
        let rest = s
            .split_once(marker)
            .map(|(_, rest)| rest)
            .ok_or_else(|| anyhow!("only GitHub repositories are supported for PR creation"))?;
        let mut parts = rest.split('/');
        let owner = parts
            .next()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| anyhow!("GitHub repo owner missing"))?;
        let name = parts
            .next()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| anyhow!("GitHub repo name missing"))?
            .trim_end_matches(".git");
        Ok(GithubRepo {
            owner: owner.to_string(),
            name: name.to_string(),
        })
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
