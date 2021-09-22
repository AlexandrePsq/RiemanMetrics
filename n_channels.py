def n_channels(dataset):
    raw = dataset.get_single_subject_data(subject=1)
    raw = raw['session_0']['run_0']
    n_channels = len(raw.ch_names)
    return n_channels