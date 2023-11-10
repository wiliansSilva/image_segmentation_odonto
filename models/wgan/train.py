import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from critic import Critic
from generator import Generator
from utils import initialize_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01 # c in paper

if __name__ == "__main__":
    transf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ])

    dataset = datasets.MNIST(root="dataset/", train=True, transforms=transf, download=True)
    #dataset = datasets.ImageFolder(root="dataset/", train=True, transforms=transf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)

    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.RMSprop(gen.parameters(), lr=LR)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LR)

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
                loss_critic = -(torch.mean(cricit_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

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

                    write_real.add_image("Real", img_grid_real, global_step=step)
                    write_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
