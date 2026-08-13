#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniforge or provide conda on PATH." >&2
  exit 1
fi

echo "This installs the isolated H100 environment and is approval-sensitive."
echo "After user approval run: conda env create -f ${project_root}/environment.h100.yml"
