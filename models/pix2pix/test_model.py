import torch
from torch import nn

from discriminator import Discriminator
from generator import Generator

def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    preds = disc(x, y)
    print(f"Discriminator shape: {preds.shape}")

    gen = Generator(3, 64)
    preds = gen(x)
    print(f"Generator shape: {preds.shape}")

if __name__ == "__main__":
    test()