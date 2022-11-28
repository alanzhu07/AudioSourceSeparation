import numpy as np

def sample_seconds(audio, sample_rate, seconds, rng=None):
    """
    audio : (np.ndarray or List[np.ndarray]) (num_channels, length)
    """

    if type(audio) is list:
        original_length = audio[0].shape[1]
    else:
        original_length = audio.shape[1]

    sample_length = sample_rate*seconds
    if original_length < sample_length:
        raise Exception("audio too short")

    if not rng:
        rng = np.random.default_rng()
    i = rng.integers(original_length-sample_length)

    if type(audio) is list:
        output = [a[:,i:i+sample_length] for a in audio]
    else:
        output = audio[:,i:i+sample_length]

    return output

def sample_frames(audio, sample_length, rng=None):

    if type(audio) is list:
        original_length = audio[0].shape[1]
    else:
        original_length = audio.shape[1]

    if original_length < sample_length:
        raise Exception("audio too short")

    if not rng:
        rng = np.random.default_rng()
    i = rng.integers(original_length-sample_length)

    if type(audio) is list:
        output = [a[:,i:i+sample_length] for a in audio]
    else:
        output = audio[:,i:i+sample_length]

    return output

def sample_tracks(tracks, sample_rate, seconds, rng=None):
    """
    tracks: List[np.ndarray()] or List[List[np.ndarray()]]

    output: (num_tracks, num_channels, sampled_length)
    """

    output = [sample_seconds(track, sample_rate, seconds, rng=rng) for track in tracks]
    output = np.array(output)

    return output

def sample_musdb_tracks(tracks, seconds, sample_targets="vocals", rng=None):
    """
    tracks: musdb.audio_classes.Track

    output: (num_tracks, num_channels, sampled_length)
    """

    if not sample_targets:
        output = []
        for track in tracks:
            track.chunk_duration = seconds
            track.chunk_start = rng.uniform(0, track.duration - track.chunk_duration)
            output.append(track.audio.T)
        output = np.array(output)

        return output
    else:
        mixtures = []
        targets = []
        for track in tracks:
            track.chunk_duration = seconds
            track.chunk_start = rng.uniform(0, track.duration - track.chunk_duration)
            mixture_sampled = track.audio.T
            target_sampled = track.targets[sample_targets].audio.T
            mixtures.append(mixture_sampled)
            targets.append(target_sampled)
        mixtures, targets = np.array(mixtures), np.array(targets)

        return mixtures, targets
    