# Data Card

## Dataset

`facies_logs_anonymized.csv`

## Purpose

This dataset is the anonymized public release used to reproduce a facies classification experiment based on well logs under blind-well validation.

## Records

- Rows: 4,253
- Wells: 6 anonymized wells
- Formations: 5 anonymized formations
- Target classes: 7 facies codes (`0`, `1`, `2`, `3`, `4`, `5`, `8`)

## Columns

- `Nombre`: anonymized well identifier
- `Formacion`: anonymized formation identifier
- `Depth`: zero-shifted depth within each well
- `NetRes`, `BVW`, `GR`, `PEF`, `PHID`, `PHIE`, `PHIN`, `ResD`, `ResM`, `ResS`, `RHOB`, `Ro`, `RT`, `SwA`, `Vshl`: petrophysical logs and derived indicators
- `Facies`: facies label

## Public Anonymization Rules

- Well names replaced with sequential labels.
- Formation names replaced with sequential labels.
- Depth shifted per well so the minimum depth is zero.
- Petrophysical values preserved.
- Facies labels preserved.

## Known Quality Issues Preserved for Reproducibility

- `PEF` is missing in 579 rows.
- `PHIE` contains 2 malformed text values that are corrected during preprocessing in the experiment pipeline.

## Modeling Notes

The public experiment excludes `PEF` from the main feature set because its missingness is concentrated in one anonymized well, which would distort blind-well validation.
