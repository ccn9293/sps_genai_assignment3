# def generate_samples(model, device, num_samples=10):
# TODO: generate num_samples points in the latent space, run the generator to construct the image, and plot the samples on a grid
#plt.show()

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_samples(model, device, num_samples=30, z_dim=100):
    model.eval()
    with torch.no_grad():
        #noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
        noise = torch.randn(num_samples, z_dim).to(device)
        fake = model(noise).detach().cpu()
    grid = make_grid(fake, nrow=int(num_samples**0.5), normalize=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Generated Samples ({num_samples})")
    plt.show()