import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_gan import get_model
from trainer_gan import train_gan

batch_size = 100

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])


def get_data_loader(data_dir='./data', batch_size=100, train=True): 
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=train)
    return loader

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # do I need this?
train_loader = get_data_loader(train=True, batch_size=batch_size)


test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) # do I need this?
test_loader = get_data_loader(train=False, batch_size=batch_size)

