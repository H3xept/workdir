//! Agent-run endpoints.

use crate::agent;
use crate::auth::AuthContext;
use crate::error::{ApiError, ApiResult};
use crate::model::{AgentRun, CreateAgentRunRequest};
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use serde_json::{json, Value};

fn run_view(run: &AgentRun) -> Value {
    json!({
        "id": run.id,
        "state": run.state.as_str(),
        "sandbox_id": run.sandbox_id,
        "template": run.template,
        "repo": run.repo,
        "prompt": run.prompt,
        "model": run.model,
        "agent": run.agent.as_str(),
        "api_key_secret": run.api_key_secret,
        "hardness": run.hardness.as_str(),
        "loop": run.r#loop,
        "github": run.github,
        "verification_result": run.verification_result,
        "branch": run.branch,
        "commit": run.commit,
        "pr_url": run.pr_url,
        "error": run.error,
        "logs_truncated": run.logs_truncated,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "finished_at": run.finished_at,
        "status_url": format!("/v1/agent-runs/{}", run.id),
        "logs_url": format!("/v1/agent-runs/{}/logs", run.id),
    })
}

fn load_owned(state: &AppState, ctx: &AuthContext, id: &str) -> ApiResult<AgentRun> {
    let run = state
        .store
        .get_agent_run(id)
        .map_err(ApiError::Internal)?
        .ok_or_else(|| ApiError::NotFound(format!("agent run {id}")))?;
    if run.org_id != ctx.org_id && !ctx.admin {
        return Err(ApiError::NotFound(format!("agent run {id}")));
    }
    Ok(run)
}

pub async fn create(
    State(state): State<AppState>,
    Extension(ctx): Extension<AuthContext>,
    Json(req): Json<CreateAgentRunRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let run = agent::start_agent_run(state, ctx, req)?;
    Ok((StatusCode::ACCEPTED, Json(run_view(&run))))
}

pub async fn list(
    State(state): State<AppState>,
    Extension(ctx): Extension<AuthContext>,
) -> ApiResult<Json<Value>> {
    let runs = state
        .store
        .list_agent_runs_for_org(&ctx.org_id)
        .map_err(ApiError::Internal)?;
    Ok(Json(json!({
        "agent_runs": runs.iter().map(run_view).collect::<Vec<_>>()
    })))
}

pub async fn get(
    State(state): State<AppState>,
    Extension(ctx): Extension<AuthContext>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let run = load_owned(&state, &ctx, &id)?;
    Ok(Json(run_view(&run)))
}

pub async fn logs(
    State(state): State<AppState>,
    Extension(ctx): Extension<AuthContext>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let run = load_owned(&state, &ctx, &id)?;
    Ok(Json(json!({
        "id": run.id,
        "state": run.state.as_str(),
        "stdout": run.stdout,
        "stderr": run.stderr,
        "diff": run.diff,
        "truncated": run.logs_truncated,
    })))
}
