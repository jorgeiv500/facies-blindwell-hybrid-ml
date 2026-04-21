from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize the facies well-log dataset.")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to the private source XLSX/CSV file.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/facies_logs_anonymized.csv"),
        help="Destination CSV path for the anonymized dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/anonymization_metadata.json"),
        help="Destination JSON path for anonymization metadata.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def build_name_map(values: list[str], prefix: str) -> dict[str, str]:
    return {value: f"{prefix}_{idx:02d}" for idx, value in enumerate(sorted(values), start=1)}


def main() -> None:
    args = parse_args()
    df = load_table(args.input_path).copy()

    original_wells = sorted(df["Nombre"].astype(str).unique().tolist())
    original_formations = sorted(df["Formacion"].astype(str).unique().tolist())
    well_map = build_name_map(original_wells, "Well")
    formation_map = build_name_map(original_formations, "Formation")

    df["Nombre"] = df["Nombre"].astype(str).map(well_map)
    df["Formacion"] = df["Formacion"].astype(str).map(formation_map)
    df["Depth"] = (
        df.groupby("Nombre")["Depth"].transform(lambda s: (pd.to_numeric(s, errors="coerce") - pd.to_numeric(s, errors="coerce").min()).round(2))
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    metadata = {
        "anonymization_steps": [
            "Well identifiers replaced with sequential labels.",
            "Formation identifiers replaced with sequential labels.",
            "Depth values shifted so each well starts at zero while preserving sampling intervals.",
            "Petrophysical measurements and facies labels preserved.",
        ],
        "n_rows": int(len(df)),
        "n_wells": int(df["Nombre"].nunique()),
        "n_formations": int(df["Formacion"].nunique()),
        "columns": list(df.columns),
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Anonymized dataset written to {args.output_path}")
    print(f"Metadata written to {args.metadata_path}")


if __name__ == "__main__":
    main()
