import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_gan import train_loader, test_loader, get_data_loader
from model_gan import get_model
from trainer_gan import train_gan
from generator import generate_samples


def main():
    z_dim=100
    epochs=10
    lr=0.00004
    device=torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader = get_data_loader('data/train', batch_size=100, train=True)
    test_loader = get_data_loader('data/test', batch_size=100, train=False)

    gan = get_model("GAN", z_dim=z_dim)
    generator = gan["Generator"]
    critic = gan["Critic"]

    criterion = nn.BCELoss()
    optimizers = {
        "Generator": optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)),
        "Critic": optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))}

    trained_model = train_gan(
        gan, train_loader, criterion, optimizers, device=device, epochs=epochs, z_dim=z_dim)

    generate_samples(trained_model["Generator"], device, num_samples=16, z_dim=z_dim)

if __name__ == "__main__":
    main()




