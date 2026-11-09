"""Figure 4B: group-mean theta and gamma scalp power topographies.

Data source: TFR_CONCATENATED_*.npy arrays from the original Figure 4A,
shape (participants, channels, frequencies, time). These arrays should
ALREADY contain baseline-normalised power (z-scores).

Individual *-power-tfr.h5 files are read ONLY for channel positions and the
condition-specific frequency and time axes. No re-normalisation is performed.

Run in a Python environment with numpy, matplotlib and MNE-Python installed.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne

# ---------------------------------------------------------------------
# 1. EDIT PATHS / SETTINGS
# ---------------------------------------------------------------------
DATA_DIR = Path(r"F:\research project\# study 1\eeg_data\set\Y")

# IMPORTANT: point this at the directory containing TFR_CONCATENATED_Y_*.npy.
# If your notebook already defines tfr_group_dir, you can instead set:
# GROUP_TFR_DIR = Path(tfr_group_dir)
GROUP_TFR_DIR = Path(r"F:\research project\# study 1\eeg_data\set\Y group\sensors\tfr") # CHANGE if needed

OUTPUT_DIR = GROUP_TFR_DIR / "Figure4B_topoplots"
REFERENCE_SUBJECT = "s1_pac_sub01"

BANDS = {
    "Theta (4–8 Hz)": (4.0, 8.0),
    "Gamma (30–80 Hz)": (30.0, 80.0),
}

WINDOWS = {
    "0–200 ms": (0.0, 0.2),
    "200–450 ms": (0.2, 0.45),
}

# Rows: FTT, De-CRAT baseline, De-CRAT adaptation.
# Each row has planning early/late and execution early/late.
CONDITIONS = [
    ("FTT", "BL", None),
    ("De-CRAT baseline", "MAIN", "baseline"),
    ("De-CRAT adaptation", "MAIN", "adaptation"),
]
STAGES = [("Planning", "plan"), ("Execution", "go")]

CMAP = "plasma"   # Same as original Figure 4A
BAND_VLIMS = {
    "Theta (4–8 Hz)": (-1.0, 3.0),
    "Gamma (30–80 Hz)": (-0.3, 1.0),
}
DPI = 300


# ---------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def naming(task, stage_code, block):
    """Return the original group-array name and individual H5 suffix."""
    stem = f"{task}_{stage_code}"
    if block is not None:
        stem += f"_{block}"
    return f"Y_{stem}", f"_{stem}_power-tfr.h5"


def read_reference_tfr(task, stage_code, block):
    """Read an individual TFR for channel metadata and exact time axis."""
    _, suffix = naming(task, stage_code, block)
    path = (
        DATA_DIR / REFERENCE_SUBJECT / "preproc" / "analysis"
        / f"{REFERENCE_SUBJECT}{suffix}"
    )
    if not path.exists():
        raise FileNotFoundError(f"Reference TFR not found: {path}")

    tfr = mne.time_frequency.read_tfrs(str(path), verbose=False)
    if isinstance(tfr, list):
        if len(tfr) != 1:
            raise ValueError(f"Expected exactly one TFR in {path}")
        tfr = tfr[0]

    # Important: do not call apply_baseline here.
    tfr = tfr.copy().pick("eeg")
    return tfr


def load_group_topographies():
    """Average pre-normalised data over participants, frequency and time."""
    topographies = {}
    reference_info = None
    reference_channels = None

    for row_label, task, block in CONDITIONS:        
        for stage_label, stage_code in STAGES:
            name, _ = naming(task, stage_code, block)
            npy_path = GROUP_TFR_DIR / f"TFR_CONCATENATED_{name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Group array not found: {npy_path}\n"
                    "Set GROUP_TFR_DIR to your existing tfr_group_dir."
                )

            tfr = read_reference_tfr(task, stage_code, block)
            freqs, times = tfr.freqs, tfr.times
            data = np.load(npy_path, mmap_mode="r")

            if data.ndim != 4:
                raise ValueError(
                    f"{npy_path.name}: expected [subjects, channels, freqs, time], "
                    f"got {data.shape}"
                )
            if data.shape[1:] != (len(tfr.ch_names), len(freqs), len(times)):
                raise ValueError(
                    f"Dimension mismatch for {name}: group {data.shape}, "
                    f"reference {(len(tfr.ch_names), len(freqs), len(times))}"
                )

            if reference_info is None:
                reference_info = tfr.info.copy()
                reference_channels = list(tfr.ch_names)
            elif tfr.ch_names != reference_channels:
                raise ValueError(
                    f"Channel order differs for {name}. Check creation of the .npy arrays."
                )

            print(f"{name:27s}  group shape={data.shape}")

            for band_label, (fmin, fmax) in BANDS.items():
                freq_mask = (freqs >= fmin - 1e-8) & (freqs <= fmax + 1e-8)
                if not freq_mask.any():
                    raise ValueError(f"No frequency samples for {band_label} in {name}")

                for window_label, (tmin, tmax) in WINDOWS.items():
                    # Non-overlapping, half-open windows [start, end).
                    time_mask = (times >= tmin) & (times < tmax)
                    if not time_mask.any():
                        raise ValueError(f"No samples for {window_label} in {name}")

                    subset = data[:, :, freq_mask, :][:, :, :, time_mask]
                    if not np.isfinite(subset).all():
                        raise ValueError(f"NaN or infinity detected: {name}, {band_label}, {window_label}")

                    # Mean across participants, selected frequencies, and time.
                    topo = np.asarray(subset.mean(axis=(0, 2, 3)))
                    topographies[(row_label, stage_label, band_label, window_label)] = topo

                    vmin, vmax = BAND_VLIMS[band_label]
                    outside = np.mean((topo < vmin) | (topo > vmax)) * 100
                    print(
                        f"  {band_label:17s} {stage_label:9s} {window_label:11s}: "
                        f"min={topo.min():+6.3f}, max={topo.max():+6.3f}, "
                        f"outside colour limits={outside:4.1f}% channels"
                    )

    return topographies, reference_info


def plot_band(topographies, info, band_label, output_stem):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.2), layout="constrained")
    windows = list(WINDOWS)
    vlim = BAND_VLIMS[band_label]
    last_image = None

    for row, (condition_label, _, _) in enumerate(CONDITIONS):
        for stage_index, (stage_label, _) in enumerate(STAGES):
            for window_index, window_label in enumerate(windows):
                col = stage_index * len(windows) + window_index
                ax = axes[row, col]
                topo = topographies[
                    (condition_label, stage_label, band_label, window_label)
                ]
                last_image, _ = mne.viz.plot_topomap(
                    topo,
                    info,
                    axes=ax,
                    cmap=CMAP,
                    vlim=vlim,
                    contours=6,
                    sensors=True,
                    show=False,
                )

                if row == 0:
                    ax.set_title(f"{stage_label}\n{window_label}", fontsize=11, pad=10)

        axes[row, 0].set_ylabel(condition_label, fontsize=12, labelpad=18)

    colorbar = fig.colorbar(
        last_image,
        ax=axes,
        location="right",
        shrink=0.83,
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Baseline-normalised power (z-score)", fontsize=11)
    fig.suptitle(f"Figure 4B — {band_label}", fontsize=15, weight="bold")

    for ext in ("png", "svg"):
        output_path = OUTPUT_DIR / f"{output_stem}.{ext}"
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print("Saved:", output_path)

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    topographies, info = load_group_topographies()
    figs = []
    for band_label, stem in [
        ("Theta (4–8 Hz)", "Figure4B_theta_topoplots"),
        ("Gamma (30–80 Hz)", "Figure4B_gamma_topoplots"),
    ]:
        figs.append(plot_band(topographies, info, band_label, stem))
    plt.show()
    return figs


if __name__ == "__main__":
    main()
