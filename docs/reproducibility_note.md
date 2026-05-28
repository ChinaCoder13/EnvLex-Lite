# Reproducibility Note

This repository is organized to reproduce the workflow reported in the EnvLex-Lite manuscript.

## Required inputs

The following files must be present in `data/`:

- `rule_library.csv`
- `environmental_cases.csv`
- `case_outputs.csv`
- `train_ids.csv`
- `val_ids.csv`
- `test_ids.csv`
- `data_dictionary.csv`

## Recommended run order

```bash
python code/run_baselines.py
python code/train_model.py
python code/evaluate_results.py
python code/run_ablation.py
python code/make_tables.py
python code/make_figures.py
```

A complete run can also be started with:

```bash
python code/run_all.py
```

## Expected outputs

The scripts write outputs into:

- `results/` for CSV tables, metrics, predictions, and ablation summaries,
- `figures/` for manuscript figures.

## Leakage-control principle

Reference-output fields, proof-state labels, corrective actions, final case-file text, and human-review labels are not used as model inputs. They are used only for supervision, validation, or evaluation.
