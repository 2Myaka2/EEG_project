from pathlib import Path
import mne


DATA_DIR = Path("data/raw")


def find_edf_files(data_dir: Path):
    return sorted(data_dir.rglob("*.edf"))


def clean_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename_dict = {}

    for ch_name in raw.ch_names:
        new_name = ch_name

        if new_name.startswith("EEG "):
            new_name = new_name.replace("EEG ", "")

        if new_name == "ECG ECG":
            new_name = "ECG"

        rename_dict[ch_name] = new_name

    raw = raw.copy().rename_channels(rename_dict)
    return raw


def main():
    edf_files = find_edf_files(DATA_DIR)

    print(f"Found EDF files: {len(edf_files)}")

    if not edf_files:
        print("No EDF files found. Check data/raw directory.")
        return

    first_file = edf_files[0]
    print(f"Reading file: {first_file}")

    raw = mne.io.read_raw_edf(first_file, preload=False, verbose=False)
    raw = clean_channel_names(raw)

    eeg_channels = [ch for ch in raw.ch_names if ch != "ECG"]

    print("\nBasic info:")
    print(f"All channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"EEG channels: {len(eeg_channels)}")
    print(f"EEG channel names: {eeg_channels}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")


if __name__ == "__main__":
    main()