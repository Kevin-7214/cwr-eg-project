# I-GATE-C Freeze and Approval Scope

- Train/Dev documents: 4047
- Reused canary features: 810
- New feature documents: 3237
- Calibration/Test documents exposed: 0/0
- Gate freeze SHA-256: `5e45b6ea927b51220fd5e1964c154f4f6621bf1a08e4841d973ad51f8d6cfd89`
- Feature scope SHA-256: `56d235c7d55bdbc89334df57b52c7e6cdc9457e5e6c53523740c22f604ec203f`
- Tensor variants: full, LOFO KGW, Unigram, Unbiased, SynthID
- Training runs: 10 exactly as frozen in `configs/intermediate_training_matrix.yaml`

Dependent tensor, training, and Dev-scoring scopes must bind the exact preceding artifact hashes. No scope may admit Calibration or Test rows.
