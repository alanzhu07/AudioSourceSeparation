import numpy as np

def sample_seconds(audio, sample_rate, seconds, rng=None):
    """
    audio (np.ndarray): (num_channels, length)
    """

    original_length = audio.shape[1]
    sample_length = sample_rate*seconds
    if original_length < sample_length:
        raise Exception("audio too short")

    if not rng:
        rng = np.random.default_rng()
    i = rng.integers(original_length-sample_length)
    output = audio[:,i:i+sample_length]

    return output

def sample_frames(audio, sample_length, rng=None):

    original_length = audio.shape[1]
    if original_length < sample_length:
        raise Exception("audio too short")

    if not rng:
        rng = np.random.default_rng()
    i = rng.integers(original_length-sample_length)
    output = audio[:,i:i+sample_length]

    return output

def sample_tracks(tracks, sample_rate, seconds, rng=None):
    """
    tracks: List[np.ndarray()]

    output: (num_tracks, num_channels, sampled_length)
    """

    output = [sample_seconds(track, sample_rate, seconds, rng=rng) for track in tracks]
    output = np.array(output)

    return output

def sample_musdb_tracks(tracks, seconds, rng=None):
    """
    tracks: musdb.audio_classes.Track

    output: (num_tracks, num_channels, sampled_length)
    """

    output = []
    for track in tracks:
        output.append(sample_seconds(track.audio.T, track.rate, seconds, rng=rng))
    output = np.array(output)

    return output
    