#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"

apt-get update
apt-get install -y \
  curl \
  ca-certificates \
  git \
  rclone \
  tmux \
  openssh-server

if ! command -v poetry >/dev/null 2>&1; then
  curl -sSL https://install.python-poetry.org | python3 -
fi

# Ensure poetry's user-local bin is on PATH for interactive shells and current session
echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.profile
export PATH="/root/.local/bin:$PATH"

mkdir -p /workspace/runs
mkdir -p /workspace/repos
mkdir -p /root/.config/rclone
mkdir -p /var/run/sshd
mkdir -p /root/.ssh

chmod 700 /root/.ssh

if [ -n "${PUBLIC_KEY:-}" ]; then
  printf "%s\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

grep -q '^PermitRootLogin ' /etc/ssh/sshd_config \
  && sed -i 's/^PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
  || echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

grep -q '^PubkeyAuthentication ' /etc/ssh/sshd_config \
  && sed -i 's/^PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
  || echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

grep -q '^PasswordAuthentication ' /etc/ssh/sshd_config \
  && sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
  || echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config

echo "[bootstrap] git: $(command -v git || true)"
echo "[bootstrap] rclone: $(command -v rclone || true)"
echo "[bootstrap] poetry: $(command -v poetry || true)"
echo "[bootstrap] python3: $(command -v python3 || true)"
echo "[bootstrap] sshd: $(command -v sshd || true)"
echo "[bootstrap] /workspace ready"

exec /usr/sbin/sshd -D -e