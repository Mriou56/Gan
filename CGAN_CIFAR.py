import torch
import torch.nn as nn
from keras.src.utils.module_utils import torchvision
from torchvision.utils import make_grid
from torch.optim import RMSprop, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Vérification du support de MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# Les classes pour l'affichage :
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Définition du générateur
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32), num_classes=10):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_shape[1] // 4
        input_dim = latent_dim + num_classes

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
            nn.ConvTranspose2d(32, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((z, label_input), dim=1)
        x = self.fc(gen_input).view(-1, 128, self.init_size, self.init_size)
        img = self.model(x)
        return img

# Définition du discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32), num_classes=10):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_shape[1] * img_shape[2])

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_img = self.label_embedding(labels).view(labels.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_img), dim=1)
        validity = self.model(d_in)
        return validity


def build_and_train_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    writer = SummaryWriter(log_dir="runs/CGAN_CIFAR10")

    latent_size = 100
    lr = 0.0002
    train_steps = 50000
    num_classes = 10

    # Initialisation des modèles
    generator = Generator(latent_dim=latent_size, img_shape=(3, 32, 32), num_classes=num_classes).to(device)
    discriminator = Discriminator(img_shape=(3, 32, 32), num_classes=num_classes).to(device)

    # Optimiseurs
    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    valid_label = 0.9
    fake_label = 0.1
    data_iter = iter(train_loader)

    # Pour mieux observer l'évolution des images
    fixed_noise = torch.randn(64, latent_size, device=device)

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

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('Loss/Generator', g_loss.item(), step)
        writer.add_scalar('Loss/Discriminator', d_loss.item(), step)

        if step % 5000 == 0:
            print(f"Step {step}/{train_steps} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
            plot_images(generator, step)

            with torch.no_grad():
                fixed_labels = torch.randint(0, num_classes, (64,), device=device)
                fake_samples = generator(fixed_noise, fixed_labels).detach().cpu()
            grid = make_grid(fake_samples, normalize=True)
            writer.add_image('Generated Images', grid, step)

    torch.save(generator.state_dict(), 'Results/CIFAR/generator_CGAN.pth')
    torch.save(discriminator.state_dict(), 'Results/CIFAR/discriminator_CGAN.pth')


# Affichage des images générées
def plot_images(generator, step):
    generator.eval()

    z = torch.randn(10, 100, device=device)
    labels = torch.arange(0, 10, device=device)
    gen_imgs = generator(z, labels).cpu().detach().numpy()

    fig, axs = plt.subplots(2, 5, figsize=(5, 2))
    fig.subplots_adjust(hspace=0.5)
    for i, ax in enumerate(axs.flatten()):
        img = (gen_imgs[i].transpose(1, 2, 0) + 1) / 2
        ax.imshow(img)
        label_name = CIFAR10_CLASSES[labels[i].item()]
        ax.set_title(label_name, fontsize=8)
        ax.axis('off')

    plt.savefig(f"Results/CIFAR/Images_CIFAR/generated_CIFAR10_CGAN_{step}.png")
    plt.close()
    generator.train()


# Fonction de test
def test(generator_path):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    z = torch.randn(16, 100, device=device)
    plot_images(generator, "test")


if __name__ == '__main__':
    build_and_train_models()
    #test("Results/MNIST/generator_CGAN.pth")
