import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


EPOCHS = 10
correct = 0
total = 0


def train_gan(model, data_loader, criterion, optimizers, device='cpu', epochs=10, z_dim=100, print_every=100, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    generator = model["Generator"].to(device)
    critic = model["Critic"].to(device)
    optimizer_g = optimizers["Generator"]
    optimizer_c = optimizers["Critic"]

    for epoch in range(epochs):
        generator.train()
        critic.train()

        running_loss_g = 0.0
        running_loss_c = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # training the critcs
            optimizer_c.zero_grad()

            real_labels = torch.ones(batch_size, 1, device=device)
            output_real = critic(real_images)
            loss_real = criterion(output_real, real_labels)

            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            output_fake = critic(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)

            
            loss_c = loss_real + loss_fake
            loss_c.backward()
            optimizer_c.step()

            # training the generator
            optimizer_g.zero_grad()
            
            output_fake_g = critic(fake_images)
            loss_g = criterion(output_fake_g, real_labels)
            loss_g.backward()
            optimizer_g.step()

            running_loss_c += loss_c.item()
            running_loss_g += loss_g.item()

            if (i+1) % print_every == 0:
                print(f"[Batch {i+1}] Critic Loss: {running_loss_c/print_every:.4f}, Generator Loss: {running_loss_g/print_every:.4f}")
                running_loss_c = 0.0
                running_loss_g = 0.0
            
            # implementing checkpoints
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
             "critic_state_dict": critic.state_dict(),
            }, os.path.join(checkpoint_dir, f"gan_epoch_{epoch+1}.pth"))

    print("GAN training complete.")
    return {"Generator": generator, "Critic": critic}