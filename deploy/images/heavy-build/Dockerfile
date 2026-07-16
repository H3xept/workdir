# workdir heavy-build curated image: native toolchains + Docker-in-Docker.
# Min shape 2 vCPU / 8 GB / 32 GB. Built into ext4 by deploy/build-image.sh.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash coreutils curl wget ca-certificates git jq socat iproute2 iputils-ping \
      build-essential python3 python3-pip python3-venv nodejs npm iptables \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
         -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && . /etc/os-release \
    && printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' \
         "${UBUNTU_CODENAME:-$VERSION_CODENAME}" "$(dpkg --print-architecture)" \
         >/etc/apt/sources.list.d/docker.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

COPY sandbox-guest-agent /usr/local/bin/sandbox-guest-agent
COPY sandbox-init /sbin/sandbox-init
RUN chmod +x /usr/local/bin/sandbox-guest-agent /sbin/sandbox-init \
    && mkdir -p /workspace /var/lib/docker /var/log
