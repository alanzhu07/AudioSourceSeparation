import torch
from utils import sample_musdb_track, slice_musdb_track_iterator
from augmentation import freq_mask
from transforms import stft

class STFTDataset(torch.utils.data.Dataset):
    def __init__(self, mixture_stft, target_stft):
        self.mixture_stft = mixture_stft
        self.target_stft = target_stft

    def __len__(self):
        return len(self.mixture_stft)

    def __getitem__(self, idx):
        return self.mixture_stft[idx], self.target_stft[idx]

class SamplingTrackDataset(torch.utils.data.Dataset):
    def __init__(self, tracks, seconds=5., target='vocals', stft=True, device='cuda', augment=False, rng=None):
        self.tracks = tracks
        self.seconds = seconds
        self.target = target
        self.rng = rng
        self.stft = stft
        self.device = device
        self.augment = augment

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        mixture, target = sample_musdb_track(self.tracks[idx], self.seconds, self.target, self.rng)
        if self.stft:
            mixture_stft = stft(torch.tensor(mixture[None], dtype=torch.float), device=self.device)[0]
            target_stft = stft(torch.tensor(target[None], dtype=torch.float), device=self.device)[0]
            if self.augment:
                mixture_stft = freq_mask(mixture_stft, F=mixture_stft.size(1), freq_dim=1, rng=self.rng)
            return mixture_stft, target_stft
        else:
            mixture = torch.tensor(mixture, dtype=torch.float, device=self.device)
            target = torch.tensor(target, dtype=torch.float, device=self.device)
            return mixture, target

class FullTrackDataset(torch.utils.data.IterableDataset):
    def __init__(self, tracks, seconds=5., target='vocals', stft=True, device='cuda'):
        self.tracks = tracks
        self.seconds = seconds
        self.target = target
        self.stft = stft
        self.device = device

    def __iter__(self):
        for track in self.tracks:
            slice_iterator = slice_musdb_track_iterator(track, self.seconds, self.target, full=False)
            for (mixture, target) in slice_iterator:
                if self.stft:
                    mixture_stft = stft(torch.tensor(mixture[None], dtype=torch.float), device=self.device)[0]
                    target_stft = stft(torch.tensor(target[None], dtype=torch.float), device=self.device)[0]
                    yield mixture_stft, target_stft
                else:
                    mixture = torch.tensor(mixture, dtype=torch.float, device=self.device)
                    target = torch.tensor(target, dtype=torch.float, device=self.device)
                    yield mixture, target