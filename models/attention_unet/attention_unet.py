import torch
from torch import nn
from torch.nn import functional as F

class AttentionBlock(nn.Module):
    def __init__(self, x_channels, g_channels):
        super().__init__()
        self.phi_g = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding="same")
        self.theta_x = nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=2)
        self.psi = nn.Conv2d(x_channels * 2, 1, kernel_size=1, padding="same")
        self.final_conv = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(x_channels)
        )

    def forward(self, x, g):
        g = self.phi_g(g)
        x1 = self.theta_x(x)

        xg = torch.cat([g, x1], 1)

        xg = F.relu(xg)
        xg = self.psi(xg)
        xg = F.sigmoid(xg)

        xg = F.interpolate(xg, size=(x.shape[2], x.shape[3]))

        y = xg * x

        return self.final_conv(y)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, features, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=kernel_size, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=kernel_size, stride=1, padding="same"),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, features, kernel_size):
        super().__init__()
        self.conv_net = DoubleConvBlock(in_channels, features, kernel_size)
        self.final_layers = nn.Sequential(
            nn.Maxpool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        s = self.conv_net(x)
        x = self.final_layers(s)

        return x, s

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.initial_layer = AttentionBlock(out_channels, in_channels)
        self.conv_net = nn.Sequential(
            nn.Dropout(0.3),
            DoubleConvBlock(in_channels, features, kernel_size)
        )

    def forward(self, x, s):
        x = self.initial_conv(x)
        x = torch.cat([x, s], 1)
        return self.conv_net(x)

class AttentionUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = 

    def initialize_weights(self):
        for m in self.module():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        pass

def test():
    g = torch.randn((1, 64, 64, 64))
    x = torch.randn((1, 128, 128, 128))

    att = AttentionBlock(128, 64)

    y = att(x, g)

    print(y.shape)

if __name__ == "__main__":
    test()