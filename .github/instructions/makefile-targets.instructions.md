---
description: "Use when creating, modifying, or reviewing Makefile train/test targets, Hydra CLI overrides, checkpoint paths, or sbatch-enabled workflows in AmbiSplice."
name: "AmbiSplice Makefile Target Rules"
applyTo: "Makefile, **/*.make, **/Makefile"
---

# Makefile Target Rules

Follow these conventions for all edits in [Makefile](../../Makefile).

- Prefer extending existing `train_*` and `test_*` patterns over inventing new command styles.
- Keep Hydra override formatting consistent with existing targets, including `+dataset.*` when adding non-default keys.
- Preserve existing `run_name` naming style (`model.dataset_variant`) unless the user explicitly asks for a different convention.
- Preserve existing checkpoint selection behavior in test targets unless the user asks to switch checkpoint logic.
- Keep conda execution style unchanged: `conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run.py ...`.
- Keep long-running training targets compatible with `sbatch_redirect` where already used.

## Safety Rules

- Do not hardcode new absolute paths.
- Do not mutate artifact directories (`checkpoints/`, `outputs/`, `lightning_logs/`) from Make targets.
- Avoid changing scheduler behavior (`sbatch`, `partition`, `time`) unless explicitly requested.

## Validation

- After Makefile edits, run the narrowest relevant command from [Makefile](../../Makefile):
  - `make help`
  - and one affected `make train_*` or `make test_*` target when feasible
- If execution is skipped (missing data, GPU, or runtime constraints), explicitly report that limitation.
