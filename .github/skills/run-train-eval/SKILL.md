---
name: run-train-eval
description: 'Run AmbiSplice training and evaluation workflows with Make targets, verify required paths and GPU settings, and summarize outcomes. Use for reproducible experiment execution and quick smoke validation.'
argument-hint: 'Target name(s) and optional constraints, for example: train_pangolinomni3_pangolinsolo123 then test_pangolinomni3_pangolinsolo123'
user-invocable: true
---

# Run Train Eval

Use this skill to execute existing AmbiSplice experiments safely and consistently.

## Use When

- You need to run or re-run an existing `make train_*` or `make test_*` workflow.
- You want a concise summary of what ran, what failed, and why.
- You need quick preflight checks before starting long jobs.

## Procedure

1. Confirm candidate targets from [Makefile](../../../Makefile), preferring existing targets over new command lines.
2. Run a fast preflight:
   - check `make help`
   - confirm referenced paths in the selected target(s) exist
   - confirm GPU request aligns with available hardware assumptions
3. Execute target(s) exactly as defined in [Makefile](../../../Makefile).
4. Capture key outputs:
   - selected target name(s)
   - whether command succeeded
   - checkpoint/output paths referenced by the run
   - first actionable error if command failed
5. Provide a compact result summary with next-step options.

## Guardrails

- Do not rewrite training/eval commands unless explicitly asked.
- Do not alter `checkpoints/`, `outputs/`, or `lightning_logs/` as part of this workflow.
- Report missing data files, unavailable GPU, or long-runtime constraints explicitly.

## References

- Runtime defaults: [config.yaml](../../../config.yaml)
- Entrypoint: [run.py](../../../run.py)
- Agent-wide repo guidance: [AGENTS.md](../../../AGENTS.md)
