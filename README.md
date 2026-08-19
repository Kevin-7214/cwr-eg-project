# CWR-EG Project

This directory is the implementation workspace for finite-open-set multi-source, multi-key watermark localization and attribution.

## Non-negotiable boundaries

- The neighboring legacy `project1` repository is read-only.
- Final spans use zero-based, half-open Unicode code-point offsets: `[char_start, char_end)`.
- `parent_id` is assigned before transformations and is the split unit.
- Train, Dev, Calibration, and Test are isolated. Formal Test is never used for fitting or threshold selection.
- The five output labels are fixed in `protocol.md`.
- Model/GPU/data-generation/training/calibration/inference/evaluation commands require explicit user approval recorded under `status/`.

## Layout

- `configs/`: frozen experiment and environment-independent configuration.
- `manifests/`: datasets, models, repositories, and generated asset records.
- `src/cwr_eg/`: implementation.
- `tests/`: CPU-only synthetic tests before the experiment gate.
- `status/`: append-only progress and approval records.
- `artifacts/`: generated experiment outputs; ignored by Git except for documentation.

## Safe pre-approval commands

Only static checks and synthetic CPU tests are permitted before the approval gate. Commands that can load a model, access CUDA, generate model text, train, calibrate, infer, evaluate, or benchmark must pass the approval guard.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m cwr_eg.cli validate-config --config configs\pilot.yaml
python -m pytest -q
python -m cwr_eg.cli status --progress status\progress.jsonl
```

The frozen intermediate manifests can be reproduced without loading a model or touching CUDA:

```powershell
python -m cwr_eg.cli validate-config --config configs\intermediate.yaml
python -m cwr_eg.cli prepare-intermediate-data `
  --legacy-corpus-dir ..\..\..\project1\data\corpus `
  --excluded-parent-manifest manifests\pilot_parents.jsonl
python -m cwr_eg.cli prepare-intermediate-canary
```

The resulting full and canary manifests are immutable inputs to the I-stage approval scopes. The canary contains 80 Train/Dev parents and 840 reusable recipes; Calibration and Test remain sealed.

## Approval workflow

1. Produce an exact fingerprint with `cwr-eg fingerprint <action> --resource-class <class> --scope-file <json-path>`.
2. Present the command, scope, expected resources, duration, outputs, and risks to the user in chat.
3. Only after explicit approval, create an ignored approval record based on `docs/approval_record.example.json`.
4. Run the matching command with `--approval`. Any action, fingerprint, resource, or expiry mismatch fails before the runtime handler imports model code.

Intermediate actions additionally include `tensorize`, `assemble-data`, `score-checkpoint`, `score-registered`, `prepare-calibration`, and `prepare-evaluation`. These remain approval-gated even when part of their work is CPU-bound because they consume experiment products or calibration/Test evidence.

Watermark key names are listed in `configs/keys.env.example`. Values are human-supplied secrets and must remain outside Git.

## Resolved Windows environment

After explicit approval for environment installation, create or repair the project-local Windows environment with:

```powershell
& .\scripts\create_windows_env.ps1 -ProjectRoot $PWD.Path
```

The script verifies fixed Miniforge and PyTorch wheel hashes, uses `conda-forge` plus official PyPI, and does not add the environment to global `PATH`. Environment installation alone does not authorize a CUDA action or model load.
