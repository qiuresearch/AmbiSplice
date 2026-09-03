---
description: "Use when editing data preparation, model development, or metrics notebooks in AmbiSplice to keep shared logic in package modules and notebook diffs minimal."
name: "AmbiSplice Notebook Editing Rules"
applyTo: "**/*.ipynb"
---

# Notebook Editing Rules

Applies to notebook workflows such as [data_prep.ipynb](../../data_prep.ipynb), [data_prep_encode.ipynb](../../data_prep_encode.ipynb), [data_prep_entex.ipynb](../../data_prep_entex.ipynb), [model_dev.ipynb](../../model_dev.ipynb), and [metrics_eval.ipynb](../../metrics_eval.ipynb).

- Prefer moving reusable logic into package modules under [AmbiSplice](../../AmbiSplice) instead of duplicating large code blocks in notebooks.
- Keep notebook edits minimal and targeted to the requested task.
- Avoid rewriting or reformatting unrelated notebook cells.
- When notebook code needs shared behavior, implement in module files first and call module functions from the notebook.

## Data and Runtime Safety

- Never modify large generated artifacts or experiment outputs unless explicitly requested.
- Avoid assumptions that all large datasets exist locally; many notebook workflows depend on pre-generated files under [data](../../data).

## Validation

- Validate module-level changes with the closest runnable command from [Makefile](../../Makefile) when practical.
- If notebook execution is skipped due to missing data, GPU constraints, or runtime cost, clearly state what was not run.
