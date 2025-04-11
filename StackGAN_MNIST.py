import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm

# Vérifie le support GPU (MPS ou CUDA sinon CPU)
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                      else "cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
latent_dim = 100
num_classes = 10
image_size = 64
high_res_size = 128
batch_size = 64
epochs = 30
lr = 0.0002
save_dir = "Results/StackGAN"
os.makedirs(save_dir, exist_ok=True)


# Générateur Stage I
class GeneratorStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = image_size // 4
        self.fc = nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((z, label_input), dim=1)
        x = self.fc(x).view(-1, 128, self.init_size, self.init_size)
        return self.model(x)

class DiscriminatorStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        label_onehot = label_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, image_size, image_size)
        d_in = torch.cat((img, label_onehot), dim=1)
        return self.model(d_in)

# === Stage II ===
class GeneratorStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, img):
        return self.upscale(img)

class DiscriminatorStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

def train_stage_I():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = GeneratorStage1().to(device)
    discriminator = DiscriminatorStage1().to(device)

    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Stage I - Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            valid = torch.ones((imgs.size(0), 1), device=device)
            fake = torch.zeros((imgs.size(0), 1), device=device)

            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z, labels)

            # === Train Generator ===
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # === Train Discriminator ===
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if (epoch + 1) % 5 == 0:
            print(f"Step {epoch}/{epochs} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
            plot_generated_images(generator, epoch, stage="Stage1")

    # Sauvegarde des poids
    torch.save(generator.state_dict(), f"{save_dir}/generator_stage1.pth")
    torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_stage1.pth")


def train_stage_II(generator_stage1):
    generator_stage2 = GeneratorStage2().to(device)
    discriminator_stage2 = DiscriminatorStage2().to(device)

    optimizer_G = Adam(generator_stage2.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator_stage2.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        generator_stage1.eval()
        loop = tqdm(range(100), desc=f"Stage II - Epoch {epoch+1}/{epochs}")
        for _ in loop:
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            z = torch.randn(batch_size, latent_dim, device=device)
            with torch.no_grad():
                low_res = generator_stage1(z, labels)

            high_res = generator_stage2(low_res)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Générateur
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator_stage2(high_res), valid)
            g_loss.backward()
            optimizer_G.step()

            # Discriminateur
            optimizer_D.zero_grad()
            real_imgs = nn.functional.interpolate(low_res, size=high_res_size, mode='bilinear', align_corners=False)
            real_loss = criterion(discriminator_stage2(real_imgs), valid)
            fake_loss = criterion(discriminator_stage2(high_res.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if (epoch + 1) % 5 == 0:
            plot_generated_images(generator_stage2, epoch, stage="Stage2", input_generator=generator_stage1)

    torch.save(generator_stage2.state_dict(), f"{save_dir}/generator_stage2.pth")
    torch.save(discriminator_stage2.state_dict(), f"{save_dir}/discriminator_stage2.pth")


def plot_generated_images(generator, epoch, stage, input_generator=None):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim, device=device)
        labels = torch.arange(0, 10, device=device)

        if input_generator:
            low_res = input_generator(z, labels)
            gen_imgs = generator(low_res)
        else:
            gen_imgs = generator(z, labels)

        img_grid = (gen_imgs + 1) / 2
        save_image(img_grid, f"{save_dir}/Images/{stage}_MNIST_Epoch{epoch}.png", nrow=5)

        fig, axs = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(img_grid[i][0].cpu(), cmap="gray")
            ax.axis("off")
        plt.suptitle(f"{stage} Generated Images - Epoch {epoch + 1}")
        plt.show()
    generator.train()

if __name__ == '__main__':
    ##train_stage_I()
    gen1 = GeneratorStage1().to(device)
    gen1.load_state_dict(torch.load("Results/StackGAN/generator_stage1.pth", map_location=device))
    train_stage_II(gen1)