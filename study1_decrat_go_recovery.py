"""RECOVERY ONLY: De-CRAT execution ROI surrogates, correct 0–0.695 s window.

First run: python study1_decrat_go_recovery.py --preflight
Then:      python -u study1_decrat_go_recovery.py --run

Requires mne, pactools, numpy, pandas, joblib. All outputs are NEW files.
The model-ready table combines one iteration across participants/conditions/ROIs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pactools.bandpass_filter import multiple_band_pass
from scipy.sparse import csr_matrix


DATA_ROOT = Path(r"F:\research project\# study 1\eeg_data\set")
SUBJECTS_DIR = Path(
    r"F:\research project\# study 1\mri_data\fs_output\freesurfer\sub_dir\Y"
)
SRC_FILE = SUBJECTS_DIR / "fsaverage_bem" / "bem" / "fsaverage-ico4-src.fif"
OUTPUT_ROOT = DATA_ROOT / "Y group" / "roi_source_analysis" / "surrogate_analysis" / "recovery_MAIN_go_200"
GROUP = "Y"
N_SURROGATES = 200
N_JOBS = 8
FS = 500.0
THETA = np.linspace(4, 8, 20)
GAMMA = np.linspace(30, 80, 20)
MIN_SHIFT = 1.0
SEED = 42

CONDITIONS = [
    ("_MAIN", "_go", "_baseline"),
    ("_MAIN", "_go", "_adaptation"),
]

ROI_NAMES = [
    "G_precentral-lh", "G_precentral-rh", "S_central-lh", "S_central-rh",
    "S_precentral-sup-part-lh", "S_precentral-sup-part-rh",
    "S_precentral-inf-part-lh", "S_precentral-inf-part-rh",
    "G_front_middle-lh", "G_front_middle-rh",
    "G_and_S_paracentral-lh", "G_and_S_paracentral-rh",
    "G_front_sup-lh", "G_front_sup-rh",
    "G_postcentral-lh", "G_postcentral-rh",
    "S_postcentral-lh", "S_postcentral-rh",
    "S_front_sup-lh", "S_front_sup-rh",
    "S_front_inf-lh", "S_front_inf-rh",
    "S_front_middle-lh", "S_front_middle-rh",
]


def find_epochs(sub: str, task: str, stage: str, block: str) -> list[Path]:
    folder = (DATA_ROOT / GROUP / sub / "preproc" / "analysis" / "source"
              / "morphed_stcs" / task / stage)
    if block:
        folder = folder / block
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing condition folder: {folder}")
    files = sorted(folder.glob("*-lh.stc"))
    if not files:
        raise FileNotFoundError(f"No left-hemisphere STC files: {folder}")
    # The user's example contains _baseline folder / _adaptation filename.
    # Fail closed: never silently mix source estimates across conditions.
    expected = re.escape(f"{sub}{task}{stage}{block}")
    pattern = re.compile(rf"^{expected}-stc-lcmv_epoch_\d+_morphed-lh\.stc$")
    bad = [p.name for p in files if not pattern.fullmatch(p.name)]
    if bad:
        raise ValueError(
            f"Folder/filename mismatch in {folder}. Example: {bad[0]}. "
            f"Expected prefix '{sub}{task}{stage}{block}'. "
            "Confirm whether these source estimates are labelled correctly."
        )
    return files


def fast_tort_pac(signal: np.ndarray) -> np.ndarray:
    """Same filters, phase bins, shifts and MI as pactools Tort estimator.

    Returns observed PAC followed by N_SURROGATES surrogate PAC values.
    """
    signal = np.asarray(signal).reshape(1, -1)
    low = multiple_band_pass(signal, FS, THETA, 2.0)
    high = multiple_band_pass(signal, FS, GAMMA, 2 * max(THETA))
    phase = np.angle(low.reshape(len(THETA), -1))
    amplitude = np.abs(high.reshape(len(GAMMA), -1))
    n = signal.size
    min_n = max(1, int(FS * MIN_SHIFT))
    if n <= 2 * min_n:
        raise ValueError("Signal too short for requested minimum shift")
    rng = np.random.RandomState(SEED)  # Identical shifts for every vertex
    shifts = rng.randint(min_n, n - min_n, size=N_SURROGATES + 1)
    shifts[0] = 0
    out = np.zeros(N_SURROGATES + 1)
    n_bins = 18
    edges = np.linspace(-np.pi, np.pi + 2 * np.finfo(phase.dtype).eps, n_bins + 1)
    for phase_row in phase:
        bins = np.digitize(phase_row, edges) - 1
        for s, shift in enumerate(shifts):
            shifted = np.roll(bins, shift)
            counts = np.bincount(shifted, minlength=n_bins)
            valid = counts > 0
            for amp in amplitude:
                sums = np.bincount(shifted, weights=amp, minlength=n_bins)
                dist = np.ones(n_bins)
                dist[valid] = sums[valid] / counts[valid]
                dist /= dist.sum()
                out[s] += np.sum(dist * np.log(dist * n_bins)) / np.log(n_bins)
    out /= len(THETA) * len(GAMMA)
    return out


def source_roi_maps(example_stc, labels):
    lh_n = len(example_stc.vertices[0])
    maps = {}
    for name in ROI_NAMES:
        lab = labels[name]
        if lab.hemi == "lh":
            idx = np.flatnonzero(np.isin(example_stc.vertices[0], lab.vertices))
        else:
            idx = np.flatnonzero(np.isin(example_stc.vertices[1], lab.vertices)) + lh_n
        if len(idx) == 0:
            raise ValueError(f"ROI has no source vertices: {name}")
        maps[name] = idx
    return maps


def process_condition(sub, task, stage, block, files, labels, adjacency):
    key = f"{sub}{task}{stage}{block}"
    folder = OUTPUT_ROOT / "conditions"
    folder.mkdir(parents=True, exist_ok=True)
    final_file = folder / f"{key}_surrogates.csv"
    checkpoint = folder / f"{key}_checkpoint.npz"
    if final_file.exists():
        print(f"Already complete: {key}", flush=True)
        return

    first = mne.read_source_estimate(str(files[0]))
    if first.data.shape[0] != adjacency.shape[0]:
        raise ValueError(f"Source/adjacency vertex count mismatch for {key}")
    roi_map = source_roi_maps(first, labels)
    roi_vertices = np.unique(np.concatenate(list(roi_map.values())))

    # Original pipeline imputes every NaN vertex using all adjacent vertices
    # whose FULL 20x20 original comodulogram contains no NaN.
    original_file = (DATA_ROOT / GROUP / sub / "preproc" / "analysis" / "source"
                     / "PAC" / f"PAC_MI_SOURCE_{sub[-5:]}{task}{stage}{block}.npy")
    if not original_file.exists():
        raise FileNotFoundError(f"Original PAC needed for NaN mask: {original_file}")
    original = np.load(original_file, mmap_mode="r")
    if original.shape != (20, adjacency.shape[0], 20):
        raise ValueError(f"Unexpected original PAC array shape: {original.shape}")
    raw_invalid = np.isnan(original).any(axis=(0, 2))
    invalid_roi = roi_vertices[raw_invalid[roi_vertices]]
    neighbours = {}
    for v in invalid_roi:
        all_neighbours = adjacency.getrow(int(v)).indices
        valid = all_neighbours[~raw_invalid[all_neighbours]]
        if len(valid) == 0:
            raise ValueError(f"No valid source neighbours for vertex {v} in {key}")
        neighbours[int(v)] = valid
    required = np.unique(np.concatenate([
        roi_vertices[~raw_invalid[roi_vertices]],
        *neighbours.values(),
    ]))
    print(f"{key}: {len(files)} epochs; {len(required)} computed vertices; "
          f"{len(invalid_roi)} imputed ROI vertices", flush=True)

    # Read each STC only once; retain just vertices needed by the analysis.
    epochs = []
    for p in files:
        # ORIGINAL _MAIN execution PAC used 0–0.695 s, unlike planning.
        stc = mne.read_source_estimate(str(p)).crop(0.0, 0.695)
        if not np.array_equal(stc.vertices[0], first.vertices[0]) or not np.array_equal(stc.vertices[1], first.vertices[1]):
            raise ValueError(f"Vertex ordering changed: {p}")
        if not np.isclose(stc.sfreq, FS):
            raise ValueError(f"Unexpected sampling rate: {stc.sfreq} for {p}")
        epochs.append(stc.data[required])
    epochs = np.stack(epochs, axis=0)
    if not np.isfinite(epochs).all():
        raise ValueError(f"Non-finite source signal in {key}; inspect before analysis")

    results = np.full((len(required), N_SURROGATES + 1), np.nan)
    if checkpoint.exists():
        with np.load(checkpoint) as cp:
            if not np.array_equal(cp["vertices"], required) or cp["values"].shape != results.shape:
                raise ValueError(f"Checkpoint settings mismatch: {checkpoint}")
            results = cp["values"].copy()
    pending = np.flatnonzero(~np.isfinite(results).all(axis=1))
    # Bounded batches: threads share source arrays without eight memory copies.
    for start in range(0, len(pending), 32):
        batch = pending[start:start + 32]
        tick = time.perf_counter()
        computed = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(fast_tort_pac)(epochs[:, int(i), :].reshape(1, -1))
            for i in batch
        )
        results[batch] = np.asarray(computed)
        if not np.isfinite(results[batch]).all():
            raise ValueError(f"Non-finite computed PAC in {key}, vertices {required[batch]}")
        np.savez(checkpoint, vertices=required, values=results)
        print(f"{key}: {np.isfinite(results).all(axis=1).sum()}/{len(required)} "
              f"vertices; batch {time.perf_counter()-tick:.1f}s", flush=True)

    # Impute with the SAME neighbours as the original analysis, separately
    # for observed and each surrogate iteration. Neighbours may lie outside ROIs.
    locations = {int(v): i for i, v in enumerate(required)}
    roi_values = {}
    for v in roi_vertices:
        if int(v) in neighbours:
            roi_values[int(v)] = np.mean(
                results[[locations[int(n)] for n in neighbours[int(v)]]], axis=0
            )
        else:
            roi_values[int(v)] = results[locations[int(v)]]

    rows = []
    for roi, vertices in roi_map.items():
        values = np.mean([roi_values[int(v)] for v in vertices], axis=0)
        for iteration in range(1, N_SURROGATES + 1):
            rows.append({"sub": sub, "task": task, "task_stage": stage,
                         "block": block, "condition": task + block,
                         "roi": roi, "surrogate_id": iteration,
                         "pac_value": values[iteration],
                         "original_pac": values[0],
                         "n_epochs": len(files),
                         "n_roi_vertices": len(vertices)})
    df = pd.DataFrame(rows)
    if len(df) != len(ROI_NAMES) * N_SURROGATES or not np.isfinite(df[["pac_value", "original_pac"]]).all().all():
        raise ValueError(f"Invalid ROI dataframe for {key}")
    # Exclusive create: never silently overwrite an existing result.
    with final_file.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    print(f"Saved {final_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--subject", help="Optional one-subject run, e.g. s1_pac_sub01")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("Choose exactly one of --preflight or --run")
    subs = sorted(p.name for p in (DATA_ROOT / GROUP).iterdir()
                  if p.is_dir() and re.fullmatch(r"s1_pac_sub\d+", p.name))
    if args.subject:
        subs = [s for s in subs if s == args.subject]
    if not subs:
        raise ValueError("No matching participant directories")
    if len(ROI_NAMES) != 24:
        raise ValueError("Expected 24 ROIs")
    entries = []
    problems = []
    for sub in subs:
        for task, stage, block in CONDITIONS:
            try:
                files = find_epochs(sub, task, stage, block)
                entries.append((sub, task, stage, block, files))
            except (FileNotFoundError, ValueError) as exc:
                problems.append(str(exc))
    print(f"Preflight: {len(subs)} participants; {len(entries)} valid condition folders; "
          f"{len(problems)} problems", flush=True)
    for message in problems[:30]:
        print("ERROR:", message, flush=True)
    if problems:
        raise SystemExit("Preflight failed. No computation started; fix paths/labels first.")
    if args.preflight:
        print("Preflight PASS. Next run --run (consider --subject first).")
        return
    labels = {lab.name: lab for lab in mne.read_labels_from_annot(
        "fsaverage_bem", parc="aparc.a2009s", subjects_dir=str(SUBJECTS_DIR))}
    missing = set(ROI_NAMES) - labels.keys()
    if missing:
        raise ValueError(f"Missing ROI labels: {sorted(missing)}")
    src = mne.read_source_spaces(str(SRC_FILE))
    # csr_matrix retains getrow() compatibility across SciPy versions.
    adjacency = csr_matrix(mne.spatial_src_adjacency(src))
    for sub, task, stage, block, files in entries:
        process_condition(sub, task, stage, block, files, labels, adjacency)
    # Write recovery-only table after every expected execution condition succeeded.
    paths = [OUTPUT_ROOT / "conditions" / f"{sub}{task}{stage}{block}_surrogates.csv"
             for sub, task, stage, block, _ in entries]
    if not all(p.exists() for p in paths):
        raise RuntimeError("Incomplete condition outputs; resume --run")
    combined = pd.concat((pd.read_csv(p, keep_default_na=False) for p in paths), ignore_index=True)
    final = OUTPUT_ROOT / ("all_subjects_MAIN_go_recovered.csv" if not args.subject
                           else f"{args.subject}_MAIN_go_recovered.csv")
    if not final.exists():
        with final.open("x", encoding="utf-8", newline="") as f:
            combined.to_csv(f, index=False)
    print(f"Finished: {len(combined)} rows; {final}", flush=True)


if __name__ == "__main__":
    main()
