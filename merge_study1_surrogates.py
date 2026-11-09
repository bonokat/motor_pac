"""Merge validated Study 1 surrogate runs and repair old ROI label ordering.

This script never edits its three source CSV files.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "upload"
OUTPUT = ROOT / "outputs" / "study1_surrogate_validation"

OLD_FILE = INPUT / "all_subjects_model7_surrogates(1).csv"
RECOVERED_FILE = INPUT / "all_subjects_MAIN_go_recovered.csv"
ORIGINAL_ROI_FILE = INPUT / "PAC_MI_SOURCE_Y_ROI.csv"

KEYS = ["sub", "task", "task_stage", "block", "roi", "surrogate_id"]
ROI_KEYS = KEYS[:-1]

# Original notebook zipped requested labels with MNE annotation-order values.
# Values on the left were written in the old CSV; those on the right are actual.
OLD_LABEL_STEMS = [
    "G_and_S_paracentral", "G_front_middle", "G_front_sup",
    "G_postcentral", "G_precentral", "S_central", "S_front_inf",
    "S_front_middle", "S_front_sup", "S_postcentral",
    "S_precentral-inf-part", "S_precentral-sup-part",
]
ACTUAL_LABEL_STEMS = [
    "S_central", "G_precentral", "S_front_inf", "S_front_middle",
    "G_and_S_paracentral", "G_front_middle", "S_precentral-inf-part",
    "S_precentral-sup-part", "S_postcentral", "S_front_sup",
    "G_postcentral", "G_front_sup",
]
LABEL_CORRECTION = {
    f"{wrong}-{hemi}": f"{actual}-{hemi}"
    for wrong, actual in zip(OLD_LABEL_STEMS, ACTUAL_LABEL_STEMS)
    for hemi in ("lh", "rh")
}


def validate_table(frame: pd.DataFrame, expected_rows: int, conditions: int) -> None:
    assert len(frame) == expected_rows, (len(frame), expected_rows)
    assert frame["sub"].nunique() == 23
    assert frame["roi"].nunique() == 24
    assert not frame.duplicated(KEYS).any()
    assert np.isfinite(frame[["pac_value", "original_pac"]].to_numpy()).all()
    counts = frame.groupby(ROI_KEYS, dropna=False).size()
    assert len(counts) == 23 * conditions * 24
    assert counts.eq(200).all()
    assert frame.groupby(ROI_KEYS, dropna=False)["original_pac"].nunique().eq(1).all()
    assert set(frame["surrogate_id"].unique()) == set(range(1, 201))


def main() -> None:
    old = pd.read_csv(OLD_FILE, keep_default_na=False)
    recovered = pd.read_csv(RECOVERED_FILE, keep_default_na=False)
    original_roi = pd.read_csv(ORIGINAL_ROI_FILE, keep_default_na=False)

    validate_table(old, 662_400, 6)
    validate_table(recovered, 220_800, 2)
    assert list(old.columns) == list(recovered.columns)
    assert set(old["sub"]) == set(recovered["sub"])

    go = (old["task"] == "_MAIN") & (old["task_stage"] == "_go")
    retained = old.loc[~go].copy()
    merged = pd.concat((retained, recovered), ignore_index=True)
    validate_table(merged, 662_400, 6)

    old_values = original_roi.copy()
    assert old_values["roi"].isin(LABEL_CORRECTION).all()
    old_values["roi_original_label"] = old_values["roi"]
    old_values["roi"] = old_values["roi"].map(LABEL_CORRECTION)
    assert len(old_values) == 3_312
    assert not old_values.duplicated(ROI_KEYS).any()

    observed = merged.drop_duplicates(ROI_KEYS)[ROI_KEYS + ["original_pac"]]
    match = old_values.merge(observed, on=ROI_KEYS, how="left", validate="one_to_one")
    assert len(match) == 3_312 and match["original_pac"].notna().all()
    abs_error = (match["pac_value"] - match["original_pac"]).abs()
    relative_error_pct = 100 * abs_error / match["original_pac"].abs().clip(lower=1e-12)
    assert relative_error_pct.max() < 0.01, relative_error_pct.max()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    merge_file = OUTPUT / "study1_surrogates_merged_validated.csv"
    corrected_file = OUTPUT / "study1_original_roi_labels_corrected.csv"
    merged.to_csv(merge_file, index=False)
    old_values.to_csv(corrected_file, index=False)

    report = (
        f"Validated source files: {OLD_FILE.name} and {RECOVERED_FILE.name}\n"
        "Unchanged: FTT plan/go and De-CRAT baseline/adaptation plan.\n"
        "Replaced: De-CRAT baseline/adaptation go (corrected 0–0.695 s window).\n"
        f"Merged rows: {len(merged):,}\n"
        f"Participants: {merged['sub'].nunique()}\n"
        "Condition-stage combinations: 6\n"
        f"ROIs: {merged['roi'].nunique()}\n"
        "Surrogate iterations: 200\n"
        "Duplicated keys: 0; nonfinite PAC: 0\n"
        "Original ROI values retained, anatomical labels repaired in a separate CSV.\n"
        f"Original PAC comparisons: {len(match):,}\n"
        f"Median relative error: {relative_error_pct.median():.8f}%\n"
        f"Maximum relative error: {relative_error_pct.max():.8f}%\n"
    )
    (OUTPUT / "validation_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(merge_file)
    print(corrected_file)


if __name__ == "__main__":
    main()
