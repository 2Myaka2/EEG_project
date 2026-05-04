from pathlib import Path
import mne


DATA_DIR = Path("data/raw")


def find_edf_files(data_dir: Path):
    return sorted(data_dir.rglob("*.edf"))


def main():
    edf_files = find_edf_files(DATA_DIR)

    print(f"Found EDF files: {len(edf_files)}")

    if not edf_files:
        print("No EDF files found. Check data/raw directory.")
        return

    first_file = edf_files[0]
    print(f"Reading file: {first_file}")

    raw = mne.io.read_raw_edf(first_file, preload=False, verbose=False)

    print("\nBasic info:")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")


if __name__ == "__main__":
    main()