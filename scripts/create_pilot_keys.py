from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets


FAMILIES = ("kgw", "unigram", "unbiased", "synthid")


def _integer_key(bits: int) -> int:
    return secrets.randbelow((1 << bits) - 2) + 1


def _synthid_key() -> list[int]:
    return secrets.SystemRandom().sample(range(1, 65536), 30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing pilot keys: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, str] = {}
    for family in FAMILIES:
        for slot in ("a", "b"):
            name = f"CWR_EG_KEY_{family.upper()}_KEY_{slot.upper()}"
            if family == "unbiased":
                values[name] = str(_integer_key(1024))
            elif family == "synthid":
                values[name] = json.dumps(_synthid_key(), separators=(",", ":"))
            else:
                values[name] = str(_integer_key(31))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        for name in sorted(values):
            handle.write(f"{name}={values[name]}\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(json.dumps({"path": str(target), "keys": len(values), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
