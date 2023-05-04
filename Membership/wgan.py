import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pytorch_fid import fid_score
from torchvision.models import inception_v3
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch.nn.functional as F
import os
from torchvision.utils import save_image
import time
import itertools

# Data paths

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001
B1=0.5
B2=0.999
BATCH_SIZE = 64
EPOCHS = 100
NUM_STEPS = 20
LATENT_DIM= 256
IMG_SIZE=32
SAVE_PER_TIMES = 10
N_CLASSES=10
CHANNELS=3

TRAIN_SIZE= 5000

NAME= "WGAN_GP2"


# function to initialize the weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(torch.nn.Module):
    def __init__(self, channels, latent_dim=LATENT_DIM, n_classes=N_CLASSES):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=latent_dim + n_classes, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))

        self.output = nn.Tanh()

    def forward(self, z, l):
        l= F.one_hot(l, N_CLASSES)
        l = l.view(l.size(0), l.size(1), 1, 1)  # Reshape l to match the dimensions of z
        x= torch.cat((z,l), dim=1)
        x = self.main_module(x)
        return self.output(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class GeneratorRes(nn.Module):
    def __init__(self,channels=CHANNELS, latent_dim=LATENT_DIM, n_classes=N_CLASSES):
        super().__init__()

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=latent_dim + n_classes, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # Adding a residual block
            ResidualBlock(in_channels=256, out_channels=256),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))

        self.output = nn.Tanh()

    def forward(self, z, l):
        l = F.one_hot(l, N_CLASSES)
        x = torch.cat((z, l.reshape(l.size()[0], l.size()[1], 1, 1)), dim=1)
        x = self.main_module(x)
        return self.output(x)



class Discriminator(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()

        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 128)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.utils.spectral_norm(nn.Linear(128, 1))

        self.ac = nn.Sequential(nn.utils.spectral_norm(nn.Linear(128, N_CLASSES)), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return self.output(x), self.ac(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 8192
        x = self.main_module(x)
        return x.view(-1, 512 * 4 * 4)


class ResidualBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlockD, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.lrelu(out)
        return out

class DiscriminatorRes(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()

        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Adding a residual block
            ResidualBlockD(in_channels=256, out_channels=256),

            # State (256x8x8)
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc = nn.Sequential(nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 128)), nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.utils.spectral_norm(nn.Linear(128, 1))

        self.ac = nn.Sequential(nn.utils.spectral_norm(nn.Linear(128, N_CLASSES)), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return self.output(x), self.ac(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 8192
        x = self.main_module(x)
        return x.view(-1, 512 * 4 * 4)

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def compute_fid_score(real_images, generated_images, inception_model, batch_size=50):
    inception_model = inception_model.to(device)
    inception_model.eval()

    real_images = F.interpolate(real_images, (299,299), mode='bilinear')
    generated_images = F.interpolate(generated_images, (299,299), mode='bilinear')

    real_dataset = TensorDataset(real_images)
    generated_dataset = TensorDataset(generated_images)

    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    generated_dataloader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

    # Calculate the activations for real and generated images
    #real_activations = get_activations(real_dataloader, inception_model)
    #generated_activations = get_activations(generated_dataloader, inception_model)

    mu1, sigma1 = get_activations(real_dataloader, inception_model)#fid_score.calculate_activation_statistics(real_activations)
    mu2, sigma2 = get_activations(generated_dataloader, inception_model)#fid_score.calculate_activation_statistics(generated_activations)

    fid = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def get_activations(dataloader, model):
    model.eval()
    activations = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            pred = model(images)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            activations.append(pred.cpu())

    pred = torch.cat(activations, dim=0)
    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class WACGAN_GP(object):
    def __init__(self, channels= 3, lr=LR, b1=B1,b2=B2, latent_dim= LATENT_DIM, n_classes= N_CLASSES, name=NAME):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(channels, latent_dim, n_classes)
        self.D = Discriminator(channels)
        #self.G.apply(weights_init_normal)
        #self.D.apply(weights_init_normal)
        self.C = channels
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

        # Check if cuda is available
        self.G.to(device)
        self.D.to(device)
        #self.auxiliary_loss.cuda()

        self.latent_dim=latent_dim
        self.n_classes= N_CLASSES
        self.learning_rate = lr
        self.b1 = b1
        self.b2 = b2
        self.name=name

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))


        self.critic_iter = 3
        self.lambda_term = 10

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(device)
        #self.inception_model = inception_v3(pretrained=True, transform_input=False)
        #self.inception_model.to(device)
        #self.inception_model = nn.Sequential(*list(self.inception_model.children())[:-1]).to(device)
        #self.inception_model.eval()
    

    def sample_images(self, num_samples, batch_size=1000):
        self.G.eval()  # Set generator to evaluation mode
        generated_images = []

        for i in range(0, num_samples, batch_size):
            with torch.no_grad():  # No need to track gradients while generating samples
                # Sample random noise
                curr_batch_size = min(batch_size, num_samples - i)
                z = torch.randn(curr_batch_size, self.latent_dim, 1, 1).to(device)

                # Sample random labels
                l = torch.randint(0, self.n_classes, (curr_batch_size,)).to(device)

                # Generate images
                curr_generated_images = self.G(z, l)
                generated_images.append(curr_generated_images.cpu())

        # Concatenate all generated images
        generated_images = torch.cat(generated_images, dim=0)
        return generated_images


    def compute_gradient_penalty(self, real_samples, fake_samples, labels):
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
        d_interpolates, _ = self.D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1).to(device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, dataloader, epochs, tic):
        best_fid = float('inf')

        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                real_imgs = imgs.to(device)
                labels = labels.to(device)

                # Train the discriminator
                for _ in range(self.critic_iter):
                    self.D.zero_grad()

                    z = torch.randn(imgs.shape[0], self.latent_dim,1,1).to(device)
                    gen_labels = torch.randint(0, self.n_classes, (imgs.shape[0],)).to(device)

                    gen_imgs = self.G(z, gen_labels)
                    real_validity, real_ac = self.D(real_imgs)
                    fake_validity, fake_ac = self.D(gen_imgs.detach())

                    gradient_penalty = self.compute_gradient_penalty(real_imgs, gen_imgs.detach(), labels)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_term * gradient_penalty

                    real_ac_loss = self.auxiliary_loss(real_ac, labels)
                    fake_ac_loss = self.auxiliary_loss(fake_ac, gen_labels)

                    d_loss = d_loss + real_ac_loss + fake_ac_loss

                    d_loss.backward()
                    self.d_optimizer.step()

                # Train the generator
                self.G.zero_grad()

                z = torch.randn(imgs.shape[0], self.latent_dim,1,1).to(device)
                gen_labels = torch.randint(0, self.n_classes, (imgs.shape[0],)).to(device)

                gen_imgs = self.G(z, gen_labels)
                fake_validity, fake_ac = self.D(gen_imgs)

                g_loss = -torch.mean(fake_validity)
                fake_ac_loss = self.auxiliary_loss(fake_ac, gen_labels)

                g_loss = g_loss + fake_ac_loss

                g_loss.backward()
                self.g_optimizer.step()
            print(f'epoch {epoch}: lossD: {(-torch.mean(real_validity)+ torch.mean(fake_validity)).cpu().item()}, lossD real {(-torch.mean(real_validity)).cpu().item()},\
                         ac loss real: {real_ac_loss.cpu().item()}, ac loss fake: {fake_ac_loss.cpu().item()}')

            if epoch % tic == 0:
                num_samples = 10000  # Choose the number of samples you want to evaluate the FID score on
                real_imgs = torch.cat([imgs for imgs, _ in itertools.islice(dataloader, num_samples // dataloader.batch_size)], 0)
                real_imgs = real_imgs[:num_samples]

                #z = torch.randn(num_samples, self.latent_dim,1,1).to(device)
                #gen_labels = torch.randint(0, self.n_classes, (num_samples,)).to(device)
                gen_imgs = self.sample_images(num_samples)
                fid_score = compute_fid_score(real_imgs.cpu().data, gen_imgs.cpu().data, self.inception_model)
                print(f"[Epoch {epoch}/{epochs}] [FID score: {fid_score}]")

                if fid_score < best_fid:
                    print(f"Improved FID score: {fid_score} (previous best: {best_fid})")
                    self.save_model()
                    best_fid= fid_score
                
                self.save_samples(gen_imgs[:10], "samples/{}/batch{}".format(NAME, epoch))

    
    def save_model(self):
        torch.save(self.G, 'models/{}_G.pth'.format(self.name))
        torch.save(self.D, 'models/{}_D.pth'.format(self.name))

    def load_model(self):
        self.D=torch.load('models/{}_D.pth'.format(self.name))
        self.G=torch.load('models/{}_G.pth'.format(self.name))
    
    def save_samples(self, imgs, pth):
        os.makedirs(pth, exist_ok=True)
        imgs=imgs.mul(0.5).add(0.5)
        for i in range(len(imgs)):
            save_image(imgs[i].data.cpu(), os.path.join(pth, 'img_{}.png'.format(i)))

if __name__=='__main__':
    TMP_DIR= sys.argv[1] #'/localscratch/lauh.66373363.0'


    random.seed(123)
    os.makedirs("samples/{}".format(NAME), exist_ok=True)
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

    training_set= Subset(full_dataset, all_indices[:TRAIN_SIZE])

    train_dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=BATCH_SIZE,  num_workers=8,
            shuffle=True
        )

    gan= WACGAN_GP()
    #gan.load_model()
    t= time.time()
    gan.train(train_dataloader,200,5)
    print('training for 100 epochs took', time.time()-t)