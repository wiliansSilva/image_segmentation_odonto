import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from discriminator import Discriminator
from generator import Generator
from utils import initialize_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

if __name__ == "__main__":
    transf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ])

    dataset = datasets.MNIST(root="dataset", train=True, transform=transf, download=True)
    #dataset = datasets.ImageFolder(root="dataset", transform=transf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(DEVICE)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(DEVICE)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
