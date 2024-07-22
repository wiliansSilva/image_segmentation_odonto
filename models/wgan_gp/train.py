import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from critic import Critic
from generator import Generator
from utils import initialize_weights, gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

if __name__ == "__main__":
    transf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ])

    dataset = datasets.MNIST(root="dataset/", train=True, transform=transf, download=True)
    #dataset = datasets.ImageFolder(root="dataset/", transform=transf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)

    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(DEVICE)

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(DEVICE)
                fake = gen(noise)
            
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=DEVICE)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            output = critic(fake).reshape(-1)
            loss_gen = -torch.mean(output)
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
