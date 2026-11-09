"""Study 1: summarize time-shifted PAC surrogates per subject x condition x ROI.

This script reads the validated merged surrogate dataset and writes one
descriptive row per participant/condition/ROI. It does NOT fit a model.

Run locally:
    python study1_summarize_surrogates.py

Optional:
    python study1_summarize_surrogates.py --input path/to/merged.csv --output path/to/summary.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"F:\research project\# study 1\eeg_data\set\Y group\roi_source_analysis\surrogate_analysis")
DEFAULT_INPUT = os.path.join(DATA_DIR, "study1_surrogates_merged_validated.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "study1_surrogate_roi_summary.csv")
GROUP_COLS = ["sub", "task", "task_stage", "block", "condition", "roi"]
N_SURROGATES = 200


def summarize(group: pd.DataFrame) -> pd.Series:
    surrogate_ids = group["surrogate_id"].to_numpy()
    if len(surrogate_ids) != N_SURROGATES or set(surrogate_ids) != set(range(1, N_SURROGATES + 1)):
        raise ValueError(f"Expected surrogate IDs 1–{N_SURROGATES} for every ROI")

    observed_values = group["original_pac"].to_numpy(dtype=float)
    if not np.allclose(observed_values, observed_values[0], atol=0, rtol=0):
        raise ValueError("original_pac varies across surrogate iterations")

    observed = float(observed_values[0])
    surrogates = group["pac_value"].to_numpy(dtype=float)
    if not np.isfinite(surrogates).all() or not np.isfinite(observed):
        raise ValueError("Non-finite PAC value detected")

    mean = float(surrogates.mean())
    sd = float(surrogates.std(ddof=1))
    if sd <= 0:
        raise ValueError("Surrogate SD is zero; z-score cannot be calculated")

    # PAC is non-negative, so an upper-tail test asks whether observed PAC
    # is greater than expected under the time-shifted null distribution.
    n_at_least_observed = int(np.count_nonzero(surrogates >= observed))
    p_upper = (n_at_least_observed + 1) / (N_SURROGATES + 1)

    # Optional two-sided descriptive p-value around the surrogate null mean.
    # Use only if a two-sided question has been justified in advance.
    null_deviations = np.abs(surrogates - mean)
    observed_deviation = abs(observed - mean)
    p_two_sided_centered = (
        int(np.count_nonzero(null_deviations >= observed_deviation)) + 1
    ) / (N_SURROGATES + 1)

    return pd.Series({
        "original_pac": observed,
        "surrogate_mean": mean,
        "surrogate_sd": sd,
        "surrogate_z": (observed - mean) / sd,
        "delta_pac": observed - mean,
        "p_upper": p_upper,
        "p_two_sided_centered_exploratory": p_two_sided_centered,
        "n_surrogates": len(surrogates),
        "n_epochs": int(group["n_epochs"].iloc[0]),
        "n_roi_vertices": int(group["n_roi_vertices"].iloc[0]),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)
    required = set(GROUP_COLS + [
        "surrogate_id", "pac_value", "original_pac", "n_epochs", "n_roi_vertices"
    ])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    if df.duplicated(GROUP_COLS + ["surrogate_id"]).any():
        raise ValueError("Duplicate surrogate IDs within a participant/condition/ROI")

    summary = (
        df.groupby(GROUP_COLS, dropna=False, sort=True)
          .apply(summarize, include_groups=False)
          .reset_index()
    )

    # The complete Study 1 analysis has 23 participants x 6 conditions x 24 ROIs.
    expected_rows = 23 * 6 * 24
    if len(summary) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summary rows, got {len(summary)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved {len(summary):,} rows to {args.output}")
    print("Minimum possible empirical p-value:", 1 / (N_SURROGATES + 1))
    print(summary[["surrogate_z", "p_upper"]].describe().to_string())


if __name__ == "__main__":
    main()
