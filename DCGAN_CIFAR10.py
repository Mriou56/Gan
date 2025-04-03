import torch
import torch.nn as nn
from torch.optim import RMSprop
from torchvision import datasets, transforms
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

from torchvision.datasets import CIFAR10

# Verification of the availability of the MPS device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Define the generator and discriminator networks
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        layer_filters = [256, 128, 64, 3]

        self.model = nn.Sequential(
            nn.Linear(latent_dim, layer_filters[0] * 8 * 8),
            nn.Unflatten(1, (layer_filters[0], 8, 8)),
            nn.BatchNorm2d(layer_filters[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_filters[0], layer_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_filters[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_filters[1], layer_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_filters[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_filters[2], layer_filters[3], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()
        kernel_size = 5
        layer_filters = [64, 128, 256, 512]

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], layer_filters[0], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(layer_filters[0], layer_filters[1], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(layer_filters[1], layer_filters[2], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(layer_filters[2], layer_filters[3], kernel_size=kernel_size, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(layer_filters[3] * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

def build_and_train_models():
    # Define the transformation to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Data parameters
    latent_size = 100
    lr = 2e-4
    decay = 6e-8
    train_steps = 40000

    # Initialize the generator and discriminator
    generator = Generator(latent_dim=latent_size).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = RMSprop(generator.parameters(), lr=lr)
    optimizer_D = RMSprop(discriminator.parameters(), lr=lr)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Training
    for step in range(train_steps):
        for real_imgs, _ in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Adversarial ground truths
            valid = torch.ones((batch_size, 1), device=device, requires_grad=False)
            fake = torch.zeros((batch_size, 1), device=device, requires_grad=False)

            # Train the generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_size).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train the discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if step % 500 == 0:
            print(f"Step {step}/{train_steps} | D loss: {d_loss.item()} | G loss: {g_loss.item()}")

            plt.figure(figsize=(2.2, 2.2))
            num_images = gen_imgs.shape[0]
            image_size = gen_imgs.shape[2]
            rows = int(math.sqrt(num_images))

            for i in range(num_images):
                plt.subplot(rows, rows, i + 1)
                image = (gen_imgs[i].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                plt.imshow(image)
                plt.axis('off')

    # Save the trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


def plot_images(generator, noise_input, show=False, step=0, model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, f"{step:05d}.png")
    gen_imgs = generator(noise_input)

    plt.figure(figsize=(2.2, 2.2))
    num_images = gen_imgs.shape[0]
    image_size = gen_imgs.shape[2]
    rows = int(math.sqrt(num_images))

    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = (gen_imgs[i].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        plt.imshow(image)
        plt.axis('off')

    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

def test(generator_path, latent_size=100, num_images=16):
    print("C'est le test")
    generator = Generator(latent_dim=latent_size).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    noise_input = torch.randn(num_images, latent_size).to(device)
    plot_images(generator, noise_input, show=True, model_name="test_outputs")

if __name__ == '__main__':
    build_and_train_models()
    test(generator_path="generator.pth", latent_size=100, num_images=16)