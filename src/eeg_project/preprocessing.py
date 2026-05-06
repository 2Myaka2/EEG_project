import mne


def preprocess_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float | None = 50.0,
) -> mne.io.BaseRaw:
    raw = raw.copy()

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False,
    )

    if notch_freq is not None:
        raw.notch_filter(
            freqs=notch_freq,
            verbose=False,
        )

    return raw


def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    duration: float = 4.0,
    overlap: float = 0.0,
) -> mne.Epochs:
    events = mne.make_fixed_length_events(
        raw,
        duration=duration,
        overlap=overlap,
    )

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=duration,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs