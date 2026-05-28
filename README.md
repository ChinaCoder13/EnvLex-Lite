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
data/rule\\\_library.csv
data/environmental\\\_cases.csv
data/case\\\_outputs.csv
data/train\\\_ids.csv
data/val\\\_ids.csv
data/test\\\_ids.csv
data/data\\\_dictionary.csv
```

The released dataset is a public research benchmark prepared from de-identified, normalized, rule-mapped attributes. The original administrative records are not included in this repository because they may contain enforcement-sensitive information.

## Code files in `code/`

The code package includes scripts such as:

```text
code/common.py
code/modeling.py
code/run\\\_baselines.py
code/train\\\_model.py
code/evaluate\\\_results.py
code/run\\\_ablation.py
code/make\\\_tables.py
code/make\\\_figures.py
code/run\\\_all.py
```

## Reproducing the workflow

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete workflow:

```bash
python code/run\\\_all.py
```

Or run steps individually:

```bash
python code/run\\\_baselines.py
python code/train\\\_model.py
python code/evaluate\\\_results.py
python code/run\\\_ablation.py
python code/make\\\_tables.py
python code/make\\\_figures.py
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

The main reported EnvLex-Lite values in the manuscript are:

|Metric|Reported value|
|-|-:|
|Accuracy|0.936|
|Macro-F1|0.921|
|Rule-matching accuracy|0.942|
|Evidence completeness score|0.914|
|Legal validation pass rate|0.967|
|Unsupported claim rate|0.028|
|Expected calibration error|0.031|

## Figure placement

Final figures in the `figures/` folder:

```text
fig1\\\_architecture.png
fig2\\\_dataset\\\_pipeline.png
fig3\\\_performance\\\_heatmap.png
fig4\\\_rule\\\_family\\\_heatmap.png
fig5a\\\_calibration\\\_curve.png
fig5b\\\_risk\\\_coverage\\\_curve.png
fig6\\\_confusion\\\_matrix.png
fig7\\\_ablation\\\_pareto.png
```



## Results folder

The `results/` folder contains CSV snapshots of tables and summary outputs aligned with the manuscript. These files can be overwritten by the code after a full run.

## Data governance

The original administrative records are not released. The public files contain only research-level attributes and do not include facility names, officer identifiers, precise location markers, administrative case numbers, or active enforcement references.

## License

See `LICENSE` for code and data-use terms.

