# CWR-EG Frozen Protocol

Status: implementation contract. Pilot compatibility remains `0.1.0-pre-experiment`; the RTX 5060 intermediate profile is `0.2.0-intermediate`.

## 1. Scientific scope

The system addresses finite-open-set watermark localization and attribution. It does not claim universal detection of arbitrary unknown watermarks. Unknown mechanisms are evaluated only through declared leave-one-family-out and proxy-unknown protocols.

## 2. Fixed output labels

Exactly one document-level label is emitted:

1. `known_scheme_known_key`
2. `known_scheme_unknown_key`
3. `suspected_unknown_scheme`
4. `uncertain`
5. `none`

No adapter may introduce a sixth semantic class. Explanatory reason codes are separate from labels.

## 3. Character coordinates

- All external spans are zero-based, half-open Unicode code-point intervals `[char_start, char_end)`.
- Raw input text is the canonical coordinate system.
- Normalization is allowed only with an explicit reversible boundary map back to the raw text.
- Byte offsets, UTF-16 code units, tokenizer offsets, and normalized-text offsets must never be exposed as final character offsets.

## 4. Split isolation

- The split unit is `parent_id`, assigned before watermarking, attack, paraphrase, translation, truncation, or mixing.
- All descendants of one `parent_id` stay in exactly one of Train, Dev, Calibration, or Test.
- Train fits learned parameters and train-only prototypes.
- Dev chooses architecture and optimization settings.
- Calibration fits thresholds, empirical nulls, validity rules, and familywise corrections.
- Test is read only after all choices and hashes are frozen.
- Pseudo-unknown families may guide Dev ablations but never enter the formal generic-gap null.

## 5. Evidence routes

- Generic route: produces `GenericResidualEvidence` without querying registered keys.
- Registered route: produces `RegisteredEvidence` through declared detector adapters and authorized key access.
- Validity route: produces `ValidityDiagnostics` from predeclared features only.
- The decision layer consumes calibrated evidence; it does not retrain encoders or detectors.

`validity_features` is the canonical field name. `parent_id` identifies split groups and is never used as a validity feature.

## 6. Training constraints

- Paired contrastive positives share the same parent content under different declared watermark/key interventions.
- Negative masks exclude identical descendants and any pair forbidden by the configured scientific contrast.
- The scheme adversary is applied only to declared positive watermarked examples.
- Gradient reversal uses `grl_scale`; its loss contribution uses the separate `scheme_adv_weight`.
- Orthogonality loss is centered and is valid only when the effective batch contains at least two valid examples; otherwise it returns zero with a diagnostic reason.
- The null prototype is fitted from Train clean/null examples only and serialized with provenance.
- Proxy scoring models are frozen/pretrained during the main CWR-EG fit unless a separate approved experiment explicitly states otherwise.
- Weak boundary labels must carry `boundary_quality=weak` and cannot be silently pooled with exact labels.

## 7. Calibration and multiplicity

- Every reported p-value names its null pool, stratum, tail, smoothing rule, and calibration hash.
- Search-aware generic p-values calibrate the entire candidate-generation and selection pipeline. `doc_generic_p` is reserved for the final document-level search-aware value.
- Registered p-values account for all tested schemes, keys, windows, and scales in their declared search family.
- The target document-level FWER is `alpha=0.01`; the predeclared acceptance upper bound is `0.015` using an exact one-sided Clopper-Pearson confidence bound.
- Calibration insufficiency, unsupported strata, invalid diagnostics, or interval-mapping failure cannot be coerced to `none`; they produce `uncertain` with reason codes.

## 8. Decision ordering

The decision layer applies calibrated, predeclared rules in this order:

1. Invalid or insufficient evidence -> `uncertain`.
2. Significant registered scheme and authorized key -> `known_scheme_known_key`.
3. Significant registered scheme but no registered key survives, with valid generic/registered gap evidence -> `known_scheme_unknown_key`.
4. Significant search-aware generic evidence with no registered scheme surviving -> `suspected_unknown_scheme`.
5. Valid evidence with no significance -> `none`.
6. Conflicts not covered above -> `uncertain`.

Document aggregation is conservative: a document inherits the highest-priority valid segment decision in the order above, except any unresolved segment-level validity conflict forces `uncertain`.

## 9. Local experiment approval

Before any model load, CUDA/GPU use, model-based data generation, attack generation, training/backpropagation, hyperparameter search, calibration, inference, metric evaluation, or benchmark, the command must verify an active approval record. An approval is valid only for the exact action, command fingerprint, resource class, and expiry described in that record. Absence or mismatch is a hard failure.

## 10. RTX 5060 intermediate profile

- The intermediate profile uses 800 parents selected with seed `20260815`, with per-source Train/Dev/Calibration/Test counts of 75/25/50/50. The 32 pilot parents are an explicit exclusion set.
- Its 8,400 recipes contain 4,000 base documents, 4,000 matched attacks, and 400 same-language, same-split mixed documents. Train mixed documents are evaluation-only and are excluded before feature extraction, leaving at most 8,250 scoring objects.
- Feature files are atomically written, individually hashed, and resumable. Tensor bundles use `sharded-v1`, with at most 16 batches per shard.
- Checkpoint scoring uses sliding 256-position windows. Token logits are projected to raw Unicode code points; `mapping_coverage` is computed before nearest-valid-token filling, and coverage below 0.98 forces `uncertain`.
- Registered tests cover every generic candidate plus the full-text interval for all authorized family/key instances. Unbiased retains its raw lower-tail statistic and uses `negate-lower-tail-statistic-v1` only as the upward-oriented search/calibration strength.
- Calibration pools are built from parent-level maxima over clean and attacked-clean descendants. The only intermediate null strata are `en:all` and `zh:all`, each with 100 independent parents.
- The three complete checkpoints are combined only by arithmetic mean of character logits. Ablations and leave-one-family-out runs remain separate reports.
- Test remains sealed through canary and Dev analysis. A new exact approval is required at every I-stage gate; no G-stage approval or fingerprint is reusable.
