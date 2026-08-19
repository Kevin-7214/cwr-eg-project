# Environment Contract

Status: environment contract. The Windows G-01 environment was resolved on 2026-08-13; Linux H100 remains a specification only.

## Supported profiles

### Windows local preparation

- OS: Windows, PowerShell.
- Python: 3.11.
- Environment manager: project-local Miniforge/Conda when available.
- GPU target: NVIDIA RTX 5060, compute capability `sm_120`.
- PyTorch target: CUDA 12.8 wheel build. The exact resolved package versions must be captured before the first approved GPU test.
- Pre-approval use: source editing, manifest preparation, static checks, and CPU-only synthetic unit tests.

#### Resolved G-01 environment

- Miniforge: `26.3.2`, installed physically under project `.miniforge/`.
- Environment: project `.conda/cwr-eg-win-py311`, Python `3.11.15`.
- Stable ASCII path alias: `E:\.cwr-eg-project-local`, verified as an NTFS junction to this project because Miniforge cannot install directly through the Chinese prefix on this volume.
- PyTorch: `2.9.1+cu128`; official wheel SHA-256 `633005a3700e81b5be0df2a7d3c1d48aced23ed927653797a3bd2b144a3aeeb6`.
- GPU observed: NVIDIA GeForce RTX 5060, driver `610.74`, compute capability `sm_120`.
- Locked package records: `manifests/environment.windows.conda-explicit.txt` and `manifests/environment.windows.pip-lock.txt`.
- G-01 executed no model download or model load.

#### RTX 5060 intermediate limits

- Only the already verified local Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct assets may be used; network model download remains disabled.
- Every approved I-stage action monitors GPU temperature, system RAM, and free space on the output volume. It stops with partial artifacts preserved at 85°C sustained for 120 seconds, RAM above 26 GiB, or free disk below 100 GiB.
- Feature extraction starts at microbatch 4 and may fall back only through 2 then 1 after an out-of-memory exception. A CUDA/driver error or OOM at microbatch 1 is terminal.
- Intermediate artifacts may add at most 30 GB. The E: volume must have at least 100 GB free before every approved action.

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
