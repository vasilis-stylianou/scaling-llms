#!/usr/bin/env bash
set -euo pipefail

IP="${1:?Usage: runpod_host.sh <IP> <PORT>}"
PORT="${2:?Usage: runpod_host.sh <IP> <PORT>}"

cat > ~/.ssh/runnning_runpod <<EOF
Host running-runpod
    HostName ${IP}
    Port ${PORT}
EOF

chmod 600 ~/.ssh/runnning_runpod

# echo "Updated SSH host: running-runpod -> ${IP}:${PORT}"
# echo
# echo "Test with:"
# echo "  ssh running-runpod"