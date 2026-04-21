# Facies Blind-Well Geologically Constrained ML

Reproducible public repository for blind-well facies classification from anonymized well logs.

This version of the repository matches the experiment currently used by the manuscript:

- leakage-aware blind-well validation
- strong tabular baselines and sequence-aware deep models
- geologically constrained sequence decoding from training-well facies transitions
- uncertainty-aware selective prediction
- reservoir-oriented diagnostics such as contact displacement and facies thickness error

## Headline Results

Average performance on the anonymized public dataset:

| Validation | Model | Accuracy | Balanced accuracy | Macro-F1 | Log-loss |
|:--|:--|--:|--:|--:|--:|
| BlindWell | `GeoConstrainedEnsemble` | 0.925 | 0.819 | 0.791 | 0.275 |
| BlindWell | `RandomForest` | 0.927 | 0.803 | 0.788 | 0.270 |
| BlindWell | `ProbabilisticEnsemble` | 0.922 | 0.816 | 0.785 | 0.256 |
| BlindWell | `GeoConstrainedHybrid` | 0.907 | 0.790 | 0.756 | 0.277 |
| RandomSplit | `GeoConstrainedHybrid` | 0.967 | 0.804 | 0.800 | 0.120 |

Main interpretation:

- random splitting remains optimistic relative to blind-well evaluation
- the strongest blind-well operating point is not a raw classifier but the geologically constrained ensemble
- the geological prior improves average blind-well macro-F1 and balanced accuracy without using blind-well labels
- deep learning remains informative as a complementary sequence-aware branch, but it is not the dominant blind-well model on this six-well dataset

Representative blind-well behavior:

- the representative case is `Well_06`, the hardest blind well for the hybrid family in this run
- in `Well_06`, `GeoConstrainedEnsemble` improves accuracy from `0.874` to `0.895` and macro-F1 from `0.599` to `0.641` relative to `ProbabilisticEnsemble`
- the learned geological smoothing strength for `Well_06` is `0.40`, while it remains `0.00` in easier folds where smoothing is unnecessary

Geological utility and selective prediction:

- `GeoConstrainedEnsemble` reduces average contact displacement from `1.283` to `1.189` depth units relative to `ProbabilisticEnsemble`
- `GeoConstrainedEnsemble` reduces average facies thickness error from `4.976` to `4.571`
- at `80%` retained coverage, `GeoConstrainedEnsemble` reaches `0.972` accuracy and `0.743` strict all-class macro-F1

## What This Repository Reproduces

The experiment compares:

- `RandomForest`
- `GradientBoosting`
- `XGBoost`
- `ProbabilisticEnsemble`
- `GeoConstrainedEnsemble`
- `CNN`
- `CNN_BiLSTM`
- `CNN_BiLSTM_Attention`
- `HybridFusion`
- `GeoConstrainedHybrid`

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
├── LICENSE
└── CITATION.cff
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

Optional path overrides:

```bash
python src/run_hybrid_experiment.py \
  --data-path data/facies_logs_anonymized.csv \
  --results-dir outputs/results \
  --figures-dir outputs/figures
```

Machine-readable outputs are written to:

- `outputs/results/metrics_summary.csv`
- `outputs/results/metrics_per_split.csv`
- `outputs/results/metrics_by_split.csv`
- `outputs/results/predictions_all.csv`
- `outputs/results/selective_curves_blindwell.csv`

## Main Figure Set

The public figure set was reduced to the figures actively used by the current manuscript and response package.

- `figure_1_workflow`: leakage-aware workflow and modeling path
- `figure_3_random_vs_blind`: optimism gap between random split and blind-well validation
- `figure_4_model_comparison`: blind-well metric comparison across raw and geologically constrained models
- `figure_5_depth_predictions`: multi-track blind-well panel with GR, PHID, RT, and facies strips
- `figure_6_uncertainty`: geo-hybrid class probabilities, confidence, entropy, margin, and MC-dropout variance
- `figure_7_shap_attention`: SHAP-based feature ranking and attention profile
- `figure_8_sequence_value`: contact displacement, thickness error, and over-segmentation
- `figure_9_selective_prediction`: coverage-risk curves for selective prediction

## Transparency Notes

- no blind-well samples are used during preprocessing, tuning, calibration, or geological smoothing estimation
- `PEF` is excluded from the main modeling workflow because it is completely missing in one well
- geological transition priors are learned only from training wells and tuned only on inner validation data
- the public dataset does not include well coordinates or structural surfaces, so the repository does not claim spatial reservoir reconstruction

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
