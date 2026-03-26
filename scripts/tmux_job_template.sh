# tmux_job_template.sh

set -euo pipefail

LOG_PATH=${log_path}
mkdir -p "$$(dirname "$$LOG_PATH")"

${work_dir_block}

export PATH="$$HOME/.local/bin:$$PATH"

get_container_env_var() {
  local key="$$1"
  tr '\0' '\n' < /proc/1/environ | grep "^$${key}=" | head -n1 | cut -d= -f2-
}

hydrate_runpod_env() {
  if [ -z "$${RUNPOD_API_KEY:-}" ]; then
    RUNPOD_API_KEY="$$(get_container_env_var RUNPOD_API_KEY)"
    export RUNPOD_API_KEY
  fi

  if [ -z "$${RUNPOD_POD_ID:-}" ]; then
    RUNPOD_POD_ID="$$(get_container_env_var RUNPOD_POD_ID)"
    export RUNPOD_POD_ID
  fi
}

stop_current_pod() {
  hydrate_runpod_env

  if [ -z "$${RUNPOD_API_KEY:-}" ]; then
    echo "[runpod] RUNPOD_API_KEY is not set" | tee -a "$$LOG_PATH"
    return 1
  fi

  if [ -z "$${RUNPOD_POD_ID:-}" ]; then
    echo "[runpod] RUNPOD_POD_ID is not set" | tee -a "$$LOG_PATH"
    return 1
  fi

  echo "[runpod] sending stop request for pod $$RUNPOD_POD_ID" | tee -a "$$LOG_PATH"

  response="$$(cat <<EOF | curl -fsS -X POST "https://api.runpod.io/graphql?api_key=$$RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d @-
{"query":"mutation { podStop(input: { podId: \"$$RUNPOD_POD_ID\" }) { id desiredStatus } }"}
EOF
  )"

  echo "$$response" | tee -a "$$LOG_PATH"
}

run_main_command() {
  ${command} 2>&1 | tee -a "$$LOG_PATH"
}

if run_main_command; then
${success_block}
else
  exit_code=$$?
${failure_block}
  exit "$$exit_code"
fi