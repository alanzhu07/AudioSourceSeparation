import torch

def stft(input,
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=torch.hann_window(4096),
    center=True):
    """
    input <torch.Tensor>: (num_samples x num_channels x length)

    output <torch.Tensor>: (num_samples x num_channels x fft_size x n_fft_frames)
    """

    num_samples, num_channels, length = input.size()
    input = input.view(-1, length)

    stft = torch.stft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, return_complex=True)
    stft = stft.view(num_samples, num_channels, *stft.size()[-2:])

    return stft

def istft(input,
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=torch.hann_window(4096),
    center=True):
    """
    input <torch.Tensor>: (num_samples x num_channels x fft_size x n_fft_frames)

    output <torch.Tensor>: (num_samples x num_channels x length)
    """

    num_samples, num_channels, fft_size, n_fft_frames = input.size()
    input = input.view(-1, fft_size, n_fft_frames)

    istft = torch.istft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, return_complex=False)
    istft = istft.view(num_samples, num_channels, -1)

    return istft