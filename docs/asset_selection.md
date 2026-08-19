# Pilot Asset Selection

## Corpora

The pilot uses four already-downloaded, hash-verified sources: C4 English, Wikipedia English, THUCNews Chinese, and Wikipedia Chinese. Eight parents are deterministically selected from each source. The existing arXiv summarization copy is excluded because its dataset card did not declare a license during the audit.

## Generators and feature models

- Qwen2.5-0.5B-Instruct: first local forward-only smoke target.
- Qwen2.5-1.5B-Instruct: local one-generation smoke target and pilot generator.
- Qwen2.5-7B-Instruct: primary H100 formal generator and proxy candidate.
- Phi-3.5-mini-instruct: cross-model generator and model-based paraphrase/round-trip translation attacker.
- BLOOM-1b7: architecture-shift generator.
- multilingual-e5-small: frozen multilingual representation view.

All are ungated candidates. Exact revisions and licenses are recorded in `manifests/model_registry.json`. Models not already present locally remain download-required and are not downloaded before an approved model phase.

## Watermark implementations

- MarkLLM is the pinned implementation source for KGW, Unigram, Unbiased, and SynthID.
- The original KGW repository is pinned as a scientific reference implementation.
- Each family has two runtime-supplied keys. Secret key values are never stored in tracked files.
- WaterSeeker is disabled until its repository license is explicit.

The exact commits and source policies are recorded in `manifests/repository_registry.json`.

## Comparison baselines

- Official GCD/AOL source is pinned but its Linux C++/pybind11 extension has not been built.
- DetectGPT and Binoculars source trees are pinned; their additional model assets are not downloaded before an approved baseline phase.
- Direct statistics, Logistic Regression, XGBoost, maximum softmax, Energy OOD, Mahalanobis, one-class SVM, prototype distance, and unstructured fusion interfaces are implemented in `cwr_eg.baselines`.
- TTP-Detect remains conditional because no public implementation was confirmed in the official paper record.

The executable/readiness status of every baseline is frozen in `configs/baselines.yaml`.

## RTX 5060 intermediate selection

The 24-hour intermediate profile deliberately narrows the asset set to files already present and hash-verified. Qwen2.5-1.5B-Instruct generates base and mixed documents; Qwen2.5-0.5B-Instruct performs paraphrase and translation-roundtrip attacks. Neither role permits a download or remote-code load.

The enabled intermediate baselines are direct statistics, Logistic Regression, maximum softmax, Energy, Mahalanobis, one-class SVM, prototype distance, generic-only, registered-only, linear evidence fusion, direct-feature MLP, and the four MarkLLM registered detectors. DetectGPT, Binoculars, BLOOM, multilingual E5, official GCD/AOL, and WaterSeeker are excluded for the exact reasons frozen in `configs/intermediate_baselines.yaml`; their exclusion is a resource/protocol boundary, not a negative performance claim.
