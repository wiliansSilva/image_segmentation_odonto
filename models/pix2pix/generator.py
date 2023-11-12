import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, in_channels: int=3, features_g: int=64):
        super().__init__()

        self.down1 = self._encoder_block(in_channels, features_g, "leaky", use_batchnorm=False)
        self.down2 = self._encoder_block(features_g, features_g * 2, "leaky")
        self.down3 = self._encoder_block(features_g * 2, features_g * 4, "leaky")
        self.down4 = self._encoder_block(features_g * 4, features_g * 8, "leaky")
        self.down5 = self._encoder_block(features_g * 8, features_g * 8, "leaky")
        self.down6 = self._encoder_block(features_g * 8, features_g * 8, "leaky")
        self.down7 = self._encoder_block(features_g * 8, features_g * 8, "leaky")

        self.bottleneck = self._encoder_block(features_g * 8, features_g * 8, activation="leaky", use_batchnorm=False)

        self.up1 = self._decoder_block(features_g * 8, features_g * 8, use_dropout=True)
        self.up2 = self._decoder_block(features_g * 16, features_g * 8, use_dropout=True)
        self.up3 = self._decoder_block(features_g * 16, features_g * 8, use_dropout=True)
        self.up4 = self._decoder_block(features_g * 16, features_g * 8, use_dropout=False)
        self.up5 = self._decoder_block(features_g * 16, features_g * 4, use_dropout=False)
        self.up6 = self._decoder_block(features_g * 8, features_g * 2, use_dropout=False)
        self.up7 = self._decoder_block(features_g * 4, features_g, use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _encoder_block(self, in_channels: int, out_channels: int, activation: str="relu", use_batchnorm: bool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
        ]

        if use_batchnorm:
            layers.append(
                nn.BatchNorm2d(out_channels)
            )

        layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels: int, out_channels: int, activation: str="relu", use_dropout: bool=False, use_batchnorm: bool=True):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        ]

        if use_batchnorm:
            layers.append(
                nn.BatchNorm2d(out_channels)
            )

        layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2))

        if use_dropout:
            layers.append(
                nn.Dropout(0.5)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        return self.final_up(torch.cat([up7, d1], 1))