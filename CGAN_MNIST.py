import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from torchvision import datasets, transforms
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Vérification du support de MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from torchvision import datasets, transforms
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Vérification du support de MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")


# Définition du générateur
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28), num_classes=10):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)  # embeddings des labels
        self.init_size = img_shape[1] // 4
        input_dim = latent_dim + num_classes  # z concaténé avec le label

        self.fc = nn.Linear(input_dim, 128 * self.init_size * self.init_size)

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
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Embedding et concaténation
        label_input = self.label_emb(labels)
        gen_input = torch.cat((z, label_input), dim=1)
        x = self.fc(gen_input).view(-1, 128, self.init_size, self.init_size)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), num_classes=10):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_shape[1] * img_shape[2])

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_img = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)
        d_in = torch.cat((img, label_img), dim=1)
        validity = self.model(d_in)
        return validity


def build_and_train_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    latent_size = 100
    lr = 0.0002
    train_steps = 50000
    num_classes = 10

    # Initialisation des modèles
    generator = Generator(latent_dim=latent_size, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # Optimiseurs
    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    valid_label = 0.9
    fake_label = 0.1
    data_iter = iter(train_loader)

    for step in tqdm(range(train_steps), desc="Training Progress"):
        try:
            real_imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_imgs, labels = next(data_iter)

        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)

        valid = torch.full((batch_size, 1), valid_label, device=device)
        fake = torch.full((batch_size, 1), fake_label, device=device)

        # Entraînement du générateur
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_size, device=device)
        gen_imgs = generator(z, labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Entraînement du discriminateur (1 fois sur 2 après un certain nombre d'étapes)
        train_d = step < 10000 or step % 2 == 0
        if train_d:
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if step % 5000 == 0:
            print(f"Step {step}/{train_steps} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
            plot_images(generator, step)

    torch.save(generator.state_dict(), 'Results/MNIST/generator_CGAN.pth')
    torch.save(discriminator.state_dict(), 'Results/MNIST/discriminator_CGAN.pth')


# Affichage des images générées
def plot_images(generator, step):
    generator.eval()

    z = torch.randn(10, 100, device=device)
    labels = torch.arange(0, 10, device=device)
    gen_imgs = generator(z, labels).cpu().detach().numpy()

    fig, axs = plt.subplots(2, 5, figsize=(5, 2))
    fig.subplots_adjust(hspace=0.5)
    for i, ax in enumerate(axs.flatten()):
        img = gen_imgs[i][0]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{labels[i].item()}", fontsize=8)
        ax.axis('off')

    plt.savefig(f"generated_MNIST_CGAN{step}.png")
    plt.close()
    generator.train()


# Fonction de test
def truc(generator_path):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    z = torch.randn(16, 100, device=device)
    plot_images(generator, "test")


if __name__ == '__main__':
    #build_and_train_models()
    truc("Results/MNIST/generator_CGAN.pth")
