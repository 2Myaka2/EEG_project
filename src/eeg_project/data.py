from pathlib import Path

import mne


def clean_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename_dict = {}

    for ch_name in raw.ch_names:
        new_name = ch_name

        if new_name.startswith("EEG "):
            new_name = new_name.replace("EEG ", "")

        if new_name == "ECG ECG":
            new_name = "ECG"

        rename_dict[ch_name] = new_name

    return raw.copy().rename_channels(rename_dict)


def load_raw_eeg(
    file_path: str | Path,
    preload: bool = True,
    drop_ecg: bool = True,
    drop_a2a1: bool = True,
    set_montage: bool = True,
) -> mne.io.BaseRaw:
    file_path = Path(file_path)

    raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
    raw = clean_channel_names(raw)

    if drop_ecg and "ECG" in raw.ch_names:
        raw.drop_channels(["ECG"])

    if drop_a2a1 and "A2-A1" in raw.ch_names:
        raw.drop_channels(["A2-A1"])

    if set_montage:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

    return raw


def get_subject_files(data_dir: str | Path, subject_id: int) -> tuple[Path, Path]:
    data_dir = Path(data_dir)

    subject_name = f"Subject{subject_id:02d}"

    rest_file = data_dir / f"{subject_name}_1.edf"
    task_file = data_dir / f"{subject_name}_2.edf"

    if not rest_file.exists():
        raise FileNotFoundError(f"Rest file not found: {rest_file}")

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    return rest_file, task_file


def load_subject_pair(
    data_dir: str | Path,
    subject_id: int,
    preload: bool = True,
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    rest_file, task_file = get_subject_files(data_dir, subject_id)

    raw_rest = load_raw_eeg(rest_file, preload=preload)
    raw_task = load_raw_eeg(task_file, preload=preload)

    return raw_rest, raw_task