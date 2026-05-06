from pathlib import Path

import pandas as pd

from eeg_project.data import load_subject_pair
from eeg_project.preprocessing import preprocess_raw, make_fixed_length_epochs
from eeg_project.features import extract_bandpower_features


def build_features_for_subject(
    data_dir: str | Path,
    subject_id: int,
    epoch_duration: float = 4.0,
) -> pd.DataFrame:
    raw_rest, raw_task = load_subject_pair(data_dir, subject_id=subject_id)

    raw_rest = preprocess_raw(raw_rest)
    raw_task = preprocess_raw(raw_task)

    epochs_rest = make_fixed_length_epochs(raw_rest, duration=epoch_duration)
    epochs_task = make_fixed_length_epochs(raw_task, duration=epoch_duration)

    features_rest = extract_bandpower_features(
        epochs=epochs_rest,
        subject_id=subject_id,
        condition="rest",
        label=0,
    )

    features_task = extract_bandpower_features(
        epochs=epochs_task,
        subject_id=subject_id,
        condition="task",
        label=1,
    )

    return pd.concat([features_rest, features_task], ignore_index=True)


def build_features_for_all_subjects(
    data_dir: str | Path,
    subject_ids: list[int] | None = None,
    epoch_duration: float = 4.0,
) -> pd.DataFrame:
    if subject_ids is None:
        subject_ids = list(range(36))

    all_features = []

    for subject_id in subject_ids:
        print(f"Processing subject {subject_id:02d}...")

        subject_features = build_features_for_subject(
            data_dir=data_dir,
            subject_id=subject_id,
            epoch_duration=epoch_duration,
        )

        all_features.append(subject_features)

    return pd.concat(all_features, ignore_index=True)