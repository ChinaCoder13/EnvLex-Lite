# EnvLex-Lite

**EnvLex-Lite** is the reproducibility repository for the manuscript:

**Rule-Constrained Machine Learning for Environmental Legal Violation Identification and Evidence-Based Case-File Generation**

The repository supports the paper's workflow for environmental legal violation identification using structured compliance evidence, inspection text, machine-readable rule objects, ML-based evidence fusion, rule-constrained validation, and reviewable case-file outputs.

## Repository structure

```text
EnvLex-Lite/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
├── code/
├── results/
├── figures/
└── docs/
```

## Folder description

|Folder|Contents|
|-|-|
|`data/`|Dataset files used by the code: rule library, case records, case outputs, split files, and data dictionary.|
|`code/`|Python scripts for model training, baseline evaluation, ablation, table creation, and figure creation.|
|`results/`|CSV snapshots of reported tables, model metrics, ablation results, and case-file outputs.|
|`figures/`|Final manuscript figures and figure-placement notes.|
|`docs/`|Data release note, rule logic note, and reproducibility guidance.|

## Data files expected in `data/`

Place the seven dataset files in the `data/` folder:

```text
data/rule_library.csv
data/environmental_cases.csv
data/case_outputs.csv
data/train_ids.csv
data/val_ids.csv
data/test_ids.csv
data/data_dictionary.csv
```

The released dataset is a public research benchmark prepared from de-identified, normalized, rule-mapped attributes. The original administrative records are not included in this repository because they may contain enforcement-sensitive information.

## Code files in `code/`

The code package includes scripts such as:

```text
code/common.py
code/modeling.py
code/run_baselines.py
code/train_model.py
code/evaluate_results.py
code/run_ablation.py
code/make_tables.py
code/make_figures.py
code/run_all.py
```

## Reproducing the workflow

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete workflow:

```bash
python code/run_all.py
```

Or run steps individually:

```bash
python code/run_baselines.py
python code/train_model.py
python code/evaluate_results.py
python code/run_ablation.py
python code/make_tables.py
python code/make_figures.py
```

## Expected manuscript-level outputs

The repository is organized to reproduce the paper's reported outputs, including:

* dataset distribution table,
* experimental setup summary,
* model-performance comparison,
* error-pattern table,
* representative case-file outputs,
* calibration curve,
* risk-coverage curve,
* confusion matrix,
* ablation Pareto map.

## Figure placement

Final figures in the `figures/` folder:

```text
Fig 3 - fig3_performance_heatmap.png
Fig 4 - fig4_rule_family_heatmap.png
Fig 5a - fig5a_calibration_curve.png
Fig 5b - fig5b_risk_coverage_curve.png
Fig 6 - fig6_confusion_matrix.png
Fig 7 - fig7_ablation_pareto.png
```



## Results folder

The `results/` folder contains CSV snapshots of tables and summary outputs aligned with the manuscript. These files can be overwritten by the code after a full run.

## Data governance

The original administrative records are not released. The public files contain only research-level attributes and do not include facility names, officer identifiers, precise location markers, administrative case numbers, or active enforcement references.

## License

See `LICENSE` for code and data-use terms.

