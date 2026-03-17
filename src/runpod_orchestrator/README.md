# runpod_orchestrator

Small infrastructure CLI for RunPod GPU training workflows.

## Commands

### Create pod

```bash
poetry run python -m runpod_orchestrator.cli create \
  --pod-spec ./src/runpod_orchestrator/pod_spec.yaml \
  --bootstrap-script ./src/runpod_orchestrator/bootstrap.sh \
  --no-reuse \
  --timeout-s 600 \
  --poll-s 5
```

### Setup existing pod

```bash
poetry run python -m runpod_orchestrator.cli setup \
  --pod-id <POD_ID> \
  --repo-url https://github.com/vasilis-stylianou/scaling-llms.git \
  --repo-dir /workspace/repos/scaling-llms \
  --identity-file ~/.ssh/runpod_key \
  --rclone-config-local ~/.config/rclone/rclone.conf
```

### Launch training in tmux

```bash
poetry run python -m runpod_orchestrator.cli train \
  --pod-id <POD_ID> \
  --cmd "poetry run python -m scaling_llms.experiments" \
  --repo-dir /workspace/repos/scaling-llms \
  --tmux-session train \
  --log-path /workspace/runs/train.log
```

### Full run workflow (create + setup + train)

```bash
poetry run python -m runpod_orchestrator.cli run \
  --pod-spec ./src/runpod_orchestrator/pod_spec.yaml \
  --bootstrap-script ./src/runpod_orchestrator/bootstrap.sh \
  --repo-url https://github.com/vasilis-stylianou/scaling-llms.git \
  --repo-dir /workspace/repos/scaling-llms \
  --cmd "poetry run python -m scaling_llms.experiments" \
  --terminate-after-launch
```

### Stop / terminate

```bash
poetry run python -m runpod_orchestrator.cli stop --pod-id <POD_ID>
poetry run python -m runpod_orchestrator.cli terminate --pod-id <POD_ID>
```

## Legacy compatibility

The old command still works:

```bash
poetry run python -m runpod_orchestrator.launcher \
  --pod-spec ./src/runpod_orchestrator/pod_spec.yaml \
  --bootstrap-script ./src/runpod_orchestrator/bootstrap.sh \
  --dry-run
```

## Smoke checks

Run pytest smoke checks:

```bash
poetry run pytest -q src/runpod_orchestrator/tests/test_cli_smoke.py
```

Run script-based smoke checks:

```bash
poetry run python -m runpod_orchestrator.smoke_check
```

Run both via `make`:

```bash
make runpod-smoke
```

## Integration tests (real RunPod resources)

These tests are opt-in and skipped by default unless `RUNPOD_E2E=1`.

Required:

```bash
export RUNPOD_E2E=1
export RUNPOD_API_KEY=your_api_key
```

Optional configuration:

```bash
export RUNPOD_TEST_POD_SPEC=./src/runpod_orchestrator/pod_spec.yaml
export RUNPOD_TEST_BOOTSTRAP_SCRIPT=./src/runpod_orchestrator/bootstrap.sh
export RUNPOD_TEST_TIMEOUT_S=900
export RUNPOD_TEST_POLL_S=5
```

Run integration lifecycle tests:

```bash
poetry run pytest -m "integration and runpod" tests/integration -vv
```
