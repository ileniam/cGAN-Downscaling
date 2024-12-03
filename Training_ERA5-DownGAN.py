#!/usr/bin/env python
# coding: utf-8

#### cGAN Downscaling

#### Imports
import os
import pandas as pd
import netCDF4 as nc
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Logger
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from scipy.stats import norm


#### PARAMETERS definition
# batch size

train_bs = 100

#### INPUT
coarse_resolution = './data/training_dataset/normalization_era5/lr_scale_lr.nc'
fine_resolution = './data/training_dataset/normalization_vhr/hr_scale_hr.nc'

#### NetCDF Custom Dataset CREATOR:

class NetCDFDataset(Dataset):
    def __init__(self, nc_dir, measure='t2m', transform=None):
        self.nc_dir = nc_dir
        self.measure = measure
        self.transform = transform
        
        # Load data
        if ( (measure).lower() == 'temp') or ( (measure).lower() == 'temperature'):
            self.data = nc.Dataset(nc_dir)['t2m']
        else:
            print(f'{measure} not recognized. Cannot retrieve the selected data. Aborting')
            self.data = None
            raise
            
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        
        single_acquisition = self.data[idx]
        
        if self.transform:
            single_acquisition = self.transform(single_acquisition)
        
        return single_acquisition


#### Dataloader Creation


input_data_transform=transforms.Compose([ transforms.ToTensor() ])

fine_data = NetCDFDataset(fine_resolution, 'temp', transform=input_data_transform)
coarse_data = NetCDFDataset(coarse_resolution, 'temp', transform=input_data_transform)

#### DataLoader creation
train_dataloader_fine = DataLoader(fine_data, batch_size=train_bs, shuffle=False)
train_dataloader_coarse = DataLoader(coarse_data, batch_size=train_bs, shuffle=False)


# Check Dimensions


random_id = 12
rows_fine = fine_data[random_id].shape[1]
columns_fine = fine_data[random_id].shape[2]

print(f"Input matrix has {rows_fine} rows and {columns_fine} columns")

random_id = 12
rows_coarse = n_covariates[random_id].shape[1]
columns_coarse = n_covariates[random_id].shape[2]

print(f"Input matrix has {rows_coarse} rows and {columns_coarse} columns")


#### MODEL SECTION

#### Discriminator


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, real_rows=680, real_columns=535, final_shape=1160):
        super(DiscriminatorNet, self).__init__()
        n_features = real_rows*real_columns 
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1160),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1160, 580),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(580, 290),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(290, n_out),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


#### Generator


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, generated_rows=58, generated_columns=80, output_rows = 680 , output_columns = 535):
        super(GeneratorNet, self).__init__()
        n_features = generated_rows*generated_columns
        n_out = output_rows*output_columns #imporre che sia un output del discriminatore
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 290),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(290, 580),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(580, 1160),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1160, n_out),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    


## ## FULL Model:

final_shape= 1160
discriminator = DiscriminatorNet(real_rows = rows_fine, real_columns = columns_fine)
generator = GeneratorNet(generated_rows = rows_coarse, generated_columns = columns_coarse, output_rows = rows_fine, output_columns = columns_fine)
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()


#### Auxiliary Functions


def images_to_vectors(images, real_rows, real_columns):
    return images.view(images.size(0), real_rows*real_columns)


def vectors_to_images(vectors, output_rows, output_columns):
    return vectors.view(vectors.size(0), 1, output_rows, output_columns)


#### Optimizations Model Params


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)


#### Weights

pixel_weight = 0.5 #(accoppiamento) con più peso
adversarial_weight = 0.2 #cambiare pesi (no accoppiamento)#0.5,0.2

#### Loss functions
d_loss = nn.BCEWithLogitsLoss()
pixel_criterion = nn.MSELoss()
adversarial_criterion = nn.BCEWithLogitsLoss()


#### Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 100


#### TRAINING


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


##### Training routines


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = d_loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = d_loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, real_data, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    # prediction = discriminator(fake_data)## codice bozza controllare 
    pixel_loss = pixel_weight * pixel_criterion(fake_data, real_data)
    # content_loss = srgan_config.content_weight * content_criterion(sr, gt)
    adversarial_loss = adversarial_weight * adversarial_criterion(discriminator(fake_data),real_data_target(real_data.size(0)))
    # error = pixel_loss + content_loss + adversarial_loss
    error = pixel_loss + adversarial_loss
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error



#### Model Run
print('inizio training')
logger = Logger(model_name='cGAN-ERA5DownGAN', data_name='EXP_ERA5DownGAN')
num_batches = len(train_dataloader_fine) 
old_error = 10**6  
for epoch in range(num_epochs):
    for n_batch,(real_batch_d, real_batch_g) in enumerate(zip(train_dataloader_fine, train_dataloader_coarse)):
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch_d, rows_fine, columns_fine))
        real_data = real_data.to(torch.float32)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        
        # Generate fake data
        generator_input = Variable(images_to_vectors(real_batch_g, rows_coarse, columns_coarse))
        generator_input = generator_input.to(torch.float32)
        fake_data = generator(generator_input)
        fake_data = fake_data.detach() 

        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                              real_data, fake_data)

        # 2. Train Generator
        generator_input = Variable(images_to_vectors(real_batch_g, rows_coarse, columns_coarse))
        generator_input = generator_input.to(torch.float32)
        # print(generator_input.shape)
        fake_data = generator(generator_input)
        # Train G
        g_error = train_generator(g_optimizer, real_data, fake_data)
       
        if abs(g_error) < abs(old_error):
            torch.save(generator.state_dict(),f'./data/models/cGAN/generator_a')
            old_error = g_error
        else: 
            torch.save(generator.state_dict(),f'./data/models/cGAN/generator_b') 
    
        # Saving d_err and g_err
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        print(n_batch, epoch)

print(epoch)



