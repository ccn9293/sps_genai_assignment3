import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 128 * 7 * 7)


        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x) # x or z?
        x = x.view(x.size(0), 128, 7, 7)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self):        
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
       
        x = self.flatten(x)
        x = self.fc(x)
        return x



def get_model(model_name, z_dim=100):
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    elif model_name == "FCNN":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
    elif model_name == "GAN": 
        generator = Generator(z_dim)
        critic = Critic()
        return {"Generator": generator, "Critic": critic}
    else:
        raise ValueError(f"Unknown model name: {model_name}")


model = get_model("GAN")
print(model)
