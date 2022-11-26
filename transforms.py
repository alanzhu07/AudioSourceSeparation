import torch

def stft(input,
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=torch.hann_window(4096),
    to_complex=True,
    center=True,
    device='cuda'):
    """
    input <torch.Tensor>: (num_samples x num_channels x length)

    output <torch.Tensor>:
        (num_samples x num_channels x fft_size x n_fft_frames) if to_complex
        else (num_samples x num_channels, 2, fft_size, n_fft_frames) 
    """

    num_samples, num_channels, length = input.size()
    input = input.view(-1, length)
    window = window.to(input.device)

    if to_complex:
        stft = torch.stft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, return_complex=True)
        stft = stft.view(num_samples, num_channels, *stft.size()[-2:])
    else:
        stft = torch.stft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, return_complex=False)
        stft = stft.movedim(-1,1)
        # stft = stft.view(num_samples*num_channels, 2, *stft.size()[-2:])

    return stft.to(device)

def istft(input,
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=torch.hann_window(4096),
    center=True,
    from_complex=True,
    num_channels=2,
    device='cuda'):
    """
    input <torch.Tensor>: (num_samples x num_channels x fft_size x n_fft_frames)

    output <torch.Tensor>: (num_samples x num_channels x length)
    """
    
    window = window.to(input.device)

    if from_complex:
        num_samples, num_channels, fft_size, n_fft_frames = input.size()
        input = input.view(-1, fft_size, n_fft_frames)
    else:
        num_samples_, _, fft_size, n_fft_frames = input.size()
        num_samples = num_samples_ // num_channels
        input = input.movedim(1,-1).contiguous()
        input = torch.view_as_complex(input)

    istft = torch.istft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, return_complex=False)
    istft = istft.view(num_samples, num_channels, -1)

    return istft.to(device)