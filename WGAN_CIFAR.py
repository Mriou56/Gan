import torch
import torch.nn as nn
from torch.optim import RMSprop
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Vérification du support de MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# -------------------- Modèles --------------------

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super().__init__()
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
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, self.init_size, self.init_size)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1)  # Pas de Sigmoid pour WGAN
        )

    def forward(self, img):
        return self.model(img)


def plot_images(generator, step, latent_dim=100):
    generator.eval()
    z = torch.randn(16, latent_dim, device=device)
    gen_imgs = generator(z).detach().cpu()

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axs.flatten()):
        img = gen_imgs[i].permute(1, 2, 0)  # CHW -> HWC
        img = (img + 1) / 2  # [-1, 1] to [0, 1] for display
        ax.imshow(img.numpy())
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"wgan_generated_{step}.png")
    plt.close()


def train_wgan():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    latent_size = 100
    lr = 0.0002
    train_steps = 50000

    # Initialisation des modèles
    generator = Generator(latent_dim=latent_size).to(device)
    discriminator = Discriminator().to(device)

    # Optimiseurs
    optimizer_G = RMSprop(generator.parameters(), lr=5e-5)
    optimizer_D = RMSprop(discriminator.parameters(), lr=5e-5)
    adversarial_loss = nn.BCELoss()

    valid_label = 0.9
    fake_label = 0.1
    data_iter = iter(train_loader)

    for step in tqdm(range(train_steps), desc="Training Progress"):
        try:
            real_imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_imgs, _ = next(data_iter)

        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        valid = torch.full((batch_size, 1), valid_label, device=device)
        fake = torch.full((batch_size, 1), fake_label, device=device)

        #if torch.rand(1).item() < 0.1:
        #    valid, fake = fake, valid

        # Entraînement du générateur
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_size, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Entraînement du discriminateur (1 fois sur 2 après un certain nombre d'étapes)
        if step < 10000:
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        elif step % 2 == 0:
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if step % 5000 == 0:
            print(f"Step {step}/{train_steps} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
            plot_images(generator, step)

    torch.save(generator.state_dict(), 'Results/CIFAR/wgan_generator.pth')
    torch.save(discriminator.state_dict(), 'Results/CIFAR/wgan_discriminator.pth')


if __name__ == '__main__':
    train_wgan()