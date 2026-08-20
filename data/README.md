# Data status

No original training dataset is present in this repository. In particular, the
previously documented `math_feedback_1000`, `train.jsonl`, and
`validation.jsonl` files were not found, so there is no recoverable set of
1,000 examples to load or migrate.

The active data artifacts are deliberately small:

- `examples/tutoring_example.v1.json` is one synthetic contract fixture.
- `../schemas/tutoring_example.v1.schema.json` is the generated canonical JSON
  Schema.
- `../evaluation/benchmarks/development_v1.jsonl` is a synthetic development
  benchmark used for regressions; it is not human-validated student data.

Every canonical example contains explicit provenance, including whether it is
synthetic. Reference solutions are plural and non-exhaustive. See
`dataset_format.md` for the storage contract.

Raw or private student data belongs in ignored locations such as `data/raw/` or
`data/private/` and must not be committed without appropriate consent,
de-identification, licensing, and governance.
