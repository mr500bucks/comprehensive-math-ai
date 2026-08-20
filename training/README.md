# Historical training records

The dated directories in this folder are preserved as historical experiment records. They are
not active installation environments and they are not currently reproducible from this
repository alone.

## Artifact status

Both `20260720` and `20260724` contain a configuration summary, a dependency list, aggregate
loss metrics, and a narrative summary. The following original artifacts were not found in the
tracked tree or elsewhere in the project directory during the 2026-08-20 inventory:

- the claimed `math_feedback_1000` dataset;
- `train.jsonl` and `validation.jsonl`;
- prompt templates and dataset formatting code;
- training, inference, and evaluation scripts;
- model adapters, checkpoints, or merged weights;
- raw trainer logs and generated evaluation examples.

The July 24 configuration also records its dataset version as `unknown`, and it does not identify
whether that run continued from the July 20 adapter or restarted from the base model.

For those reasons, both runs are classified as **legacy / partially unreproducible**. The original
measurements have not been edited or backfilled. See `legacy_artifact_status.json` for fingerprints
of the files that are available.

New runtime and development dependencies are managed at the repository root. Do not install the
dated `requirements.txt` files: they describe incomplete historical environments and include an
obsolete Transformers version.

