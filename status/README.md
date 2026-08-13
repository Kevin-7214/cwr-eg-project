# Status Records

- `progress.jsonl` is append-only during normal execution. Each line records an immediate task state transition and evidence.
- `approvals/` stores machine-readable approval records. No approval is inferred from project creation or a general request to prepare experiments.
- `GATE-LOCAL-EXPERIMENT` remains waiting until the user explicitly approves a concrete experiment request in chat.
