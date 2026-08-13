# Legacy `project1` Read-only Reuse Audit

Audit date: 2026-08-13. The legacy project is evidence and compatibility reference only; it is not modified or imported as the new public API.

| Legacy module | Reuse mode | CWR-EG boundary |
|---|---|---|
| `statistics.py` | Formula and direction reference | New code owns Unicode character coordinates and search-family accounting. |
| `calibration.py` | Empirical-tail reference | New bundle calibrates generic, registered, and gap maxima for the complete search. |
| `candidates.py` | Interval-operation reference | New candidate engine consumes raw-character evidence and returns raw Unicode intervals. |
| `metrics.py` | Metric-name reference | New evaluation clusters bootstrap by `parent_id` and separates `uncertain` from `none`. |
| `corpus.py`, `synthetic.py` | Manifest pattern | New preparation assigns and splits `parent_id` before any transformation. |
| `watermark.py` | Registered-detector concept | MarkLLM adapters declare source, revision, key access, tail direction, and applicability. |
| `attacks.py` | Attack taxonomy | New recipes apply matched attacks and carry boundary-label quality. |
| `joint.py`, `modeling.py`, `optimization.py` | Historical baseline only | Their registered-score fusion is not reused as generic unknown evidence. |

## Verified legacy state

- Five corpus JSONL files matched their declared byte sizes and SHA-256 values.
- Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct weight files matched their declared byte sizes and SHA-256 values.
- All 23 legacy unit tests passed from the legacy working directory with pytest cache and Python bytecode writes disabled.

The machine-readable evidence is in `manifests/asset_audit.json`.
