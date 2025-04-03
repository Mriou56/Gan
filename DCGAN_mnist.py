import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from torchvision import datasets, transforms
import os
import math
import matplotlib.pyplot as plt
import numpy as np

# Vérification du support de MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")


# Définition du générateur
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.init_size = img_shape[1] // 4
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, self.init_size, self.init_size)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def build_and_train_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True  # Optimized DataLoader
    )

    latent_size = 100
    lr = 2e-4
    decay = 6e-8
    train_steps = 5000

    # Initialisation des modèles
    generator = Generator(latent_dim=latent_size).to(device)
    discriminator = Discriminator().to(device)

    # Optimiseurs
    optimizer_G = RMSprop(generator.parameters(), lr=lr)
    optimizer_D = RMSprop(discriminator.parameters(), lr=lr)
    adversarial_loss = nn.BCELoss()

    for step in range(train_steps):
        print(step)
        for real_imgs, _ in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            valid = torch.ones((batch_size, 1), device=device, requires_grad=False)
            fake = torch.zeros((batch_size, 1), device=device, requires_grad=False)

            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_size).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if step % 500 == 0:
            print(f"Step {step}/{train_steps} | D loss: {d_loss.item()} | G loss: {g_loss.item()}")
            plot_images(generator, step)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


# Affichage des images générées
def plot_images(generator, step):
    generator.eval()
    z = torch.randn(16, 100, device=device)
    gen_imgs = generator(z).cpu().detach().numpy()

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axs.flatten()):
        img = gen_imgs[i][0]
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.savefig(f"generated_{step}.png")
    plt.close()


# Fonction de test
def test(generator_path):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    z = torch.randn(16, 100, device=device)
    plot_images(generator, "test")


if __name__ == '__main__':
    build_and_train_models()
    test("generator.pth")
