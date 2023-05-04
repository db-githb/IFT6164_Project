import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import torch.nn.functional as F
import os
from torchvision.utils import save_image
import time
import itertools
from wgan import *
import time

lr_rampdown_length = 0.25
lr_rampup_length = 0.05
initial_lr = 0.01
TRAIN_SIZE=5000
TMP_DIR= sys.argv[1]#'/localscratch/lauh.66574686.0/'#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE=32


def find_latent_vector(generator, target_image, latent_dim, n_classes, device, num_iterations=5000, initial_lr=initial_lr):
    target_image = target_image.to(device)
    
    # Randomly initialize the latent vector and label
    z = torch.randn(1, latent_dim).to(device).requires_grad_(True)
    label = torch.randint(0, n_classes, (1,)).to(device)

    optimizer = optim.Adam([z], lr=lr)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    for i in range(num_iterations):
        t = i / num_iterations
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()
        generated_image = generator(z, label)
        loss = loss_fn(generated_image, target_image)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iteration {i}: Loss: {loss.item()}")

    return z.detach().cpu()


def find_latent_vectors_batch(generator, target_images, target_labels, latent_dim, device, num_iterations=5000, initial_lr=initial_lr, batch_size=32):
    target_images = target_images.to(device)
    target_labels = target_labels.to(device)
    batch_size = target_images.shape[0]
    
    # Randomly initialize the latent vectors
    z = torch.randn(batch_size, latent_dim,1,1).to(device).requires_grad_(True)

    optimizer = optim.Adam([z], lr=initial_lr)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    for i in range(num_iterations):
        t = i / num_iterations
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()
        generated_images = generator(z, target_labels)
        loss = loss_fn(generated_images, target_images)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iteration {i}: Loss: {loss.item()}")

    return z.detach().cpu()


def save_images(target_image, reconstructed_image, folder, index):
    target_folder = os.path.join(folder, f"image_{index}")
    os.makedirs(target_folder, exist_ok=True)
    
    target_image = target_image.mul(0.5).add(0.5)
    save_image(target_image, os.path.join(target_folder, "target_image.png"))

    reconstructed_image = reconstructed_image.mul(0.5).add(0.5)
    save_image(reconstructed_image, os.path.join(target_folder, "reconstructed_image.png"))


def find_latent_vectors_for_dataset(generator, dataset, latent_dim, device, num_iterations=5000, initial_lr=initial_lr, batch_size=32, save_folder="results"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_vectors = []

    for index, (images, labels) in enumerate(dataloader):
        z_batch = find_latent_vectors_batch(generator, images, labels, latent_dim, device, num_iterations, initial_lr, batch_size=batch_size)
        latent_vectors.extend(z_batch.numpy())

        with torch.no_grad():
            reconstructed_images = generator(z_batch.to(device), labels.to(device)).detach().cpu()

        for i in range(images.shape[0]):
            save_images(images[i], reconstructed_images[i], save_folder, index*batch_size + i)
            np.savez(os.path.join(save_folder, f"image_{index*batch_size + i}", "latent_vector.npz"), latent_vector=z_batch[i].numpy())

    np.savez(os.path.join(save_folder, "all_latent_vectors.npz"), latent_vectors=np.array(latent_vectors))


random.seed(42)
os.makedirs("z_member5k", exist_ok=True)
os.makedirs("z_non_member5k", exist_ok=True)


print(os.path.join(TMP_DIR, "cifar-10-python.tar.gz"))
full_dataset= datasets.CIFAR10(
            os.path.join(TMP_DIR, "cifar-10-python.tar.gz"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            ),
        )

all_indices= [i for i in range(len(full_dataset))]
random.shuffle(all_indices)

training_set= Subset(full_dataset, all_indices[TRAIN_SIZE-2000:TRAIN_SIZE])
non_member_set= Subset(full_dataset, all_indices[TRAIN_SIZE:TRAIN_SIZE+2000])


gan= WACGAN_GP(name= 'WGAN_GP2')
gan.load_model()

#generator= torch.load('models/WGAN_GP_25k_G_100.pth')
find_latent_vectors_for_dataset(gan.G, training_set, latent_dim=256, device=device, num_iterations=5000, initial_lr=initial_lr, batch_size=256, save_folder="z_member5k2")

find_latent_vectors_for_dataset(gan.G, non_member_set, latent_dim=256, device=device, num_iterations=5000, initial_lr=initial_lr, batch_size=256, save_folder="z_non_member5k2")

