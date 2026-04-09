#!/bin/bash
set -e

mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Clone repo if not present
# REPO_DIR=/workspace/repos/scaling-llms
# REPO_URL=${REPO_URL:-"https://github.com/vasilis-stylianou/scaling-llms.git"}

# if [ -d "$REPO_DIR/.git" ]; then
#     echo "[entrypoint] repo exists, pulling latest..."
#     git -C "$REPO_DIR" pull
#     echo "[entrypoint] pull done"
# else
#     if [ -d "$REPO_DIR" ]; then
#         echo "[entrypoint] stale directory found, removing..."
#         cd /  # <-- get out of the directory before deleting it
#         rm -rf "$REPO_DIR"
#     fi
#     echo "[entrypoint] cloning repo..."
#     git clone "$REPO_URL" "$REPO_DIR"
#     echo "[entrypoint] clone done"
# fi

# RunPod injects the pod's public key via this env var at container start
if [ -n "${RUNPOD_PUBLIC_KEY}" ]; then
    printf '%s\n' "${RUNPOD_PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[entrypoint] SSH key injected"
else
    echo "[entrypoint] WARNING: RUNPOD_PUBLIC_KEY not set"
fi

# Write DATABASE_URL to .env for the application to use
if [ -n "${DATABASE_URL}" ]; then
    printf 'DATABASE_URL=%s\n' "${DATABASE_URL}" > /workspace/repos/scaling-llms/.env
    chmod 600 /workspace/repos/scaling-llms/.env
    echo "[entrypoint] .env written"
else
    echo "[entrypoint] WARNING: DATABASE_URL not set"
fi

# Write rclone config if RCLONE_CONF_B64 is set (base64-encoded content of rclone.conf)
if [ -n "${RCLONE_CONF_B64}" ]; then
    mkdir -p /root/.config/rclone
    if printf '%s' "${RCLONE_CONF_B64}" | base64 -d > /root/.config/rclone/rclone.conf; then
        chmod 600 /root/.config/rclone/rclone.conf
        echo "[entrypoint] rclone config written"
    else
        echo "[entrypoint] WARNING: failed to decode RCLONE_CONF_B64"
        rm -f /root/.config/rclone/rclone.conf
    fi
else
    echo "[entrypoint] WARNING: RCLONE_CONF_B64 not set"
fi

# Start SSH daemon in the background
service ssh start

# Run whatever CMD was passed
exec "$@"