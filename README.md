# scaling-llms

A research codebase for studying compute-optimal scaling laws in large language models, focusing on parameter–data tradeoffs under realistic training constraints.

Supports controlled large-scale experiments (100M–1B parameters, 10B–80B tokens) with reproducible pipelines, experiment tracking, and IsoFLOP-based analysis of loss scaling. Includes infrastructure for evaluating techniques such as muP for stable hyperparameter transfer across model sizes.

This is an active work in progress.

## System Overview

- Distributed training on RunPod GPU clusters
- Experiment tracking with Neon Postgres (runs, metrics, metadata)
- Artifact storage and synchronization via Cloudflare R2 (rclone)
- Reproducible experiment pipelines with config-driven execution

## Repository Structure

- `src/scaling_llms/` — model definitions, training loop, dataloaders, and evaluation
- `src/runpod_orchestrator/` — orchestration layer for provisioning RunPod GPU instances and executing reproducible training jobs via SSH
- `configs/` — experiment and runtime configurations
- `scripts/` — entrypoints for running experiments, locally and remotely
