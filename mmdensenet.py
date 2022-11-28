import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdensenet import MDenseNetBackbone, DenseBlock

"""
Reference: Multi-scale Multi-band DenseNets for Audio Source Separation
See https://arxiv.org/abs/1706.09588
"""

EPS = 1e-12
class MMDenseNet(nn.Module):
    """
    Multi-scale DenseNet
    """
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle','high'], sections=[380,644,1025],
        scale=(2,2),
        dilated=False, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=False,
        norm_final=True, nonlinear_final='relu',
        depth_final=None,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        self.bands = bands
        self.sections = sections
        self.bandsplit = BandSplit(sections)

        net = {}
        for band in bands:
            net[band] = MDenseNetBackbone(in_channels, num_features, growth_rate, kernel_size, scale=scale, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        
        net['full'] = MDenseNetBackbone(in_channels, num_features, growth_rate, kernel_size, scale=scale, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
        self.net = nn.ModuleDict(net)

        self.relu2d = nn.ReLU()

        # _in_channels = max([growth_rate[band][-1] for band in bands]) + growth_rate['full'][-1]
        _in_channels = growth_rate[-1]*2
        self.dense_block = DenseBlock(_in_channels, growth_rate_final, kernel_size_final, dilated=dilated_final, depth=depth_final, norm=norm_final, nonlinear=nonlinear_final, eps=eps)
        self.norm2d = nn.BatchNorm2d(growth_rate_final, eps=eps)
        self.conv2d = nn.Conv2d(growth_rate_final, in_channels, (1,1), stride=(1, 1))
        self.relu2d = nn.ReLU()

        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.scale = scale
        self.dilated, self.norm, self.nonlinear = dilated, norm, nonlinear
        self.depth = depth

        self.growth_rate_final = growth_rate_final
        self.kernel_size_final = kernel_size_final
        self.dilated_final = dilated_final
        self.depth_final = depth_final
        self.norm_final, self.nonlinear_final = norm_final, nonlinear_final

        self.eps = eps

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            output (batch_size, in_channels, n_bins, n_frames)
        """
        x = self.bandsplit(input)

        x_bands = []
        for band, x_band in zip(self.bands, x):
            x_bands.append(self.net[band](x_band))
        x_bands = torch.cat(x_bands, dim=2)
        x_full = self.net['full'](input)
        x = torch.cat([x_bands, x_full], dim=1)

        x = self.dense_block(x)
        x = self.norm2d(x)
        x = self.conv2d(x)
        x = self.relu2d(x)

        _, _, _, n_frames = x.size()
        _, _, _, n_frames_in = input.size()
        padding_width = n_frames - n_frames_in
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        output = F.pad(x, (-padding_left, -padding_right))

        return output

    def get_config(self):
        config = {
            'in_channels': self.in_channels, 'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'kernel_size': self.kernel_size,
            'bands': self.bands, 'sections': self.sections,
            'scale': self.scale,
            'dilated': self.dilated, 'norm': self.norm, 'nonlinear': self.nonlinear,
            'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'kernel_size_final': self.kernel_size_final,
            'dilated_final': self.dilated_final,
            'depth_final': self.depth_final,
            'norm_final': self.norm_final, 'nonlinear_final': self.nonlinear_final,
            'eps': self.eps
        }

        return config

    @classmethod
    def build_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        in_channels = config['in_channels']

        num_features = config['num_features']
        growth_rate = config['growth_rate']
        kernel_size = config['kernel_size']

        bands, sections = config['bands'], config['sections']
        scale = config['scale']
        dilated = config['dilated']
        norm = config['norm']
        nonlinear = config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['final']['growth_rate']
        kernel_size_final = config['final']['kernel_size']
        dilated_final = config['final']['dilated']
        depth_final = config['final']['depth']
        norm_final, nonlinear_final = config['final']['norm'], config['final']['nonlinear']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            eps=eps
        )

        return model

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels, num_features = config['in_channels'], config['num_features']
        growth_rate = config['growth_rate']

        kernel_size = config['kernel_size']
        bands, sections = config['bands'], config['sections']
        scale = config['scale']

        dilated, norm, nonlinear = config['dilated'], config['norm'], config['nonlinear']
        depth = config['depth']

        growth_rate_final = config['growth_rate_final']
        kernel_size_final = config['kernel_size_final']
        dilated_final = config['dilated_final']
        depth_final = config['depth_final']
        norm_final, nonlinear_final = config['norm_final'] or True, config['nonlinear_final']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            depth_final=depth_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

class BandSplit(nn.Module):
    def __init__(self, sections, dim=2):
        super().__init__()

        self.sections = sections
        self.dim = dim

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output: tuple of (batch_size, in_channels, sections[0], n_frames), ... (batch_size, in_channels, sections[-1], n_frames), where sum of sections is equal to n_bins
        """
        return torch.split(input, self.sections, dim=self.dim)