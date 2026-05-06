import numpy as np
import pandas as pd
from scipy.signal import welch
import mne


FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def compute_bandpower(
    signal: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> float:
    freqs, psd = welch(
        signal,
        fs=sfreq,
        nperseg=min(len(signal), int(sfreq * 2)),
    )

    freq_mask = (freqs >= fmin) & (freqs < fmax)

    if not np.any(freq_mask):
        return np.nan

    band_power = np.trapezoid(psd[freq_mask], freqs[freq_mask])
    return band_power


def extract_bandpower_features(
    epochs: mne.Epochs,
    subject_id: int,
    condition: str,
    label: int,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if freq_bands is None:
        freq_bands = FREQ_BANDS

    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    rows = []

    for epoch_idx, epoch_data in enumerate(data):
        row = {
            "subject_id": subject_id,
            "epoch_id": epoch_idx,
            "condition": condition,
            "label": label,
        }

        for ch_idx, ch_name in enumerate(ch_names):
            signal = epoch_data[ch_idx]

            for band_name, (fmin, fmax) in freq_bands.items():
                feature_name = f"{ch_name}_{band_name}_power"
                row[feature_name] = compute_bandpower(
                    signal=signal,
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                )

        rows.append(row)

    return pd.DataFrame(rows)