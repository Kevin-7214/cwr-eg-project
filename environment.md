# Environment Contract

Status: pre-experiment environment specification. No environment installation is implied by this file.

## Supported profiles

### Windows local preparation

- OS: Windows, PowerShell.
- Python: 3.11.
- Environment manager: project-local Miniforge/Conda when available.
- GPU target: NVIDIA RTX 5060, compute capability `sm_120`.
- PyTorch target: CUDA 12.8 wheel build. The exact resolved package versions must be captured before the first approved GPU test.
- Pre-approval use: source editing, manifest preparation, static checks, and CPU-only synthetic unit tests.

### Linux formal experiment

- Target: one NVIDIA H100 80GB.
- Python: 3.11.
- CUDA-enabled PyTorch environment defined separately from the Windows lock.
- Driver, CUDA runtime, scheduler, storage quota, and outbound-network policy remain human-supplied configuration.

## Isolation rules

- Do not modify the legacy `project1` environment.
- Do not install packages into system Python.
- Keep Windows and H100 lock files separate.
- Save `python --version`, package lock, GPU inventory, Git commit, config hash, and asset hashes in every approved run directory.
- Secrets and access tokens are never stored in Git-tracked files.

## Approval-sensitive actions

The following require explicit user approval before execution: environment installation that loads/tests CUDA, model download or load, GPU probe through a framework, data generation, training, calibration, inference, evaluation, or performance benchmarking. Pure package metadata resolution and configuration authoring do not constitute experiment execution, but installation is deferred until the user approves the stated command and resource impact.
