import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

# ===== Generator =====
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3*64*64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 3, 64, 64)
        return img


# ===== Create folders =====
def create_folders(base_path):
    for folder in ["0_0", "0_1", "1_0", "1_1"]:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)


# ===== Generate images =====
def generate_images(generator, save_path, num_images=200):
    generator.eval()
    
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, 100)
            fake_img = generator(z)

            # Normalize to [0,1]
            fake_img = (fake_img + 1) / 2

            # randomly assign group
            folder = ["0_0", "0_1", "1_0", "1_1"][i % 4]

            vutils.save_image(
                fake_img,
                os.path.join(save_path, folder, f"img_{i}.png")
            )


# ===== MAIN =====
if __name__ == "__main__":
    save_path = "/content/synthetic_data"

    create_folders(save_path)

    G = Generator()

    generate_images(G, save_path, num_images=400)

    print("Synthetic images generated!")