import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 1, 0),
            self._block(features_g * 4, features_g * 2, 4, 1, 0),
            nn.ConvTranspose2d(features_g * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x);
        return self.net(x)