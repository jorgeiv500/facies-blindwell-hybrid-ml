# Facies Blind-Well Hybrid ML

Reproducible repository for blind-well facies classification from anonymized well logs. The repository contains:

- an anonymized version of the dataset used in the study
- a leakage-aware experiment pipeline with tabular, sequence-aware, and hybrid models
- exported metrics and figures generated from the public anonymized data

## Headline Results

Average performance on the anonymized public dataset:

| Validation | Best model by macro-F1 | Accuracy | Macro-F1 | Balanced accuracy |
|:--|:--|--:|--:|--:|
| BlindWell | `RandomForest` | 0.927 | 0.788 | 0.803 |
| BlindWell | `ProbabilisticEnsemble` | 0.922 | 0.785 | 0.816 |
| BlindWell | `HybridFusion` | 0.907 | 0.756 | 0.790 |
| BlindWell | `CNN_BiLSTM_Attention` | 0.818 | 0.596 | 0.637 |
| RandomSplit | `HybridFusion` | 0.967 | 0.800 | 0.804 |

Interpretation:

- random splitting is optimistic compared with blind-well evaluation
- strong tree ensembles remain the most reliable cross-well baselines on this dataset
- the attention-based deep branch improves over simpler sequence models but does not dominate the tabular baselines under blind-well testing
- the hybrid branch is complementary rather than universally superior

## What This Repository Reproduces

The experiment compares:

- `RandomForest`
- `GradientBoosting`
- `XGBoost`
- a validation-weighted probabilistic ensemble
- `1D-CNN`
- `CNN_BiLSTM`
- `CNN_BiLSTM_Attention`
- a probability-level hybrid fusion between the ensemble and the attention-based deep model

Two validation protocols are reported:

- `RandomSplit`: optimistic 70/30 reference split
- `BlindWell`: leave-one-well-out evaluation across unseen wells

## Repository Layout

```text
.
├── data/
│   ├── facies_logs_anonymized.csv
│   ├── anonymization_metadata.json
│   └── DATA_CARD.md
├── outputs/
│   ├── figures/
│   └── results/
├── src/
│   ├── anonymize_dataset.py
│   └── run_hybrid_experiment.py
├── docs/
│   └── RESULTS_SUMMARY.md
├── requirements.txt
└── README.md
```

## Data Anonymization

The public dataset preserves the numerical well-log measurements and facies labels while masking potentially identifying metadata:

- well identifiers were replaced with sequential labels such as `Well_01`
- formation names were replaced with sequential labels such as `Formation_01`
- depth was shifted so each well starts at zero while preserving vertical spacing

The original private source file is not included in the repository.

## Reproduction

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full experiment:

```bash
python src/run_hybrid_experiment.py
```

Outputs will be written to:

- `outputs/results`
- `outputs/figures`

## Rebuild the Public Dataset

If you have legitimate access to the private source spreadsheet, you can regenerate the public dataset with:

```bash
python src/anonymize_dataset.py --input-path /path/to/private_source.xlsx
```

## Transparency Notes

- The experiment never uses blind-well samples during preprocessing, tuning, or probability fusion.
- `PEF` is excluded from the main modeling workflow because it is completely missing in one well.
- The public repository is intended to make the computational workflow auditable and rerunnable from anonymized data.

## Tested Environment

The published outputs were generated with:

- Python 3.13.5
- pandas 2.2.3
- numpy 2.2.6
- scikit-learn 1.6.1
- matplotlib 3.10.0
- seaborn 0.13.2
- shap 0.51.0
- torch 2.10.0
- xgboost 3.2.0
- openpyxl 3.1.5
