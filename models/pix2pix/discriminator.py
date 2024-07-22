import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        # times 2 because it receives the input image and the output image concatenated along channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                self._cnn_block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def _cnn_block(self, in_channels: int, out_channels: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)