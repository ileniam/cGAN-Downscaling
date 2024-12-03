#!/usr/bin/env python
# coding: utf-8

#### Ilenia Manco
#### cGAN Downscaling
#### TEST

from scipy import stats
import mpl_scatter_density
import os
import pandas as pd
import netCDF4 as nc
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Logger
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision
import numpy as np



#### PARAMETERS Definition
#### Batch size
test_bs = 100


#### DATA SECTION

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


# # Generator

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, generated_rows=58, generated_columns=80, output_rows = 680 , output_columns = 535):
        super(GeneratorNet, self).__init__()
        n_features = generated_rows*generated_columns
        n_out = output_rows*output_columns
        
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
    


#### Auxiliary Functions


def images_to_vectors(images, real_rows, real_columns):
    return images.view(images.size(0), real_rows*real_columns)


def vectors_to_images(vectors, output_rows, output_columns):
    return vectors.view(vectors.size(0), 1, output_rows, output_columns)


#### TEST Run

coarse_resolution_test = './data/test_dataset/normalization_vhr/lr_scale.nc'
fine_resolution_test = '.data/test_dataset/VHR_2001-2005.nc'

#### Data Transform Tensor
input_data_transform=transforms.Compose([ transforms.ToTensor() ])

coarse_data_test = NetCDFDataset(coarse_resolution_test, 'temp', transform=input_data_transform)
fine_data_test = NetCDFDataset(fine_resolution_test, 'temp', transform=input_data_transform)

#### DAtaLoader Creation
test_dataloader_coarse = DataLoader(coarse_data_test, batch_size=test_bs, shuffle=False)
test_dataloader_fine = DataLoader(fine_data_test, batch_size=test_bs, shuffle=False)


#### Check Dataset Shape


random_id = 12
rows_test_coarse = n_predictands_test[random_id].shape[1]
columns_test_coarse = n_predictands_test[random_id].shape[2]


random_id = 12
rows_test_fine = n_predictands_fine[random_id].shape[1]
columns_test_fine = n_predictands_fine[random_id].shape[2]


print(f"Input matrix has {rows_test_coarse} rows and {columns_test_coarse} columns")
print(f"Input matrix has {rows_test_fine} rows and {columns_test_fine} columns")


#### Model Generator Upload

state_dict = torch.load('./data/models/cGAN/generator_a')
model = GeneratorNet(generated_rows = rows_test_coarse, generated_columns = columns_test_coarse, output_rows = rows_test_fine, output_columns = columns_test_fine)
model.load_state_dict(state_dict)
net = model.eval()
net


#### Denormalization
#### INPUT TRAINING HR
hr_path = xr.open_dataset('.data/training_dataset/VHR_1990-2000.nc')

#### HR min-max
max_hr = hr_path.t2m.max()
min_hr = hr_path.t2m.min()
max_hr = float(max_hr)
min_hr = float(min_hr)

#### Normalization Function
def rescale_data (data,max_hr,min_hr):
    data = ((max_hr-min_hr)*data+min_hr+max_hr)/2
    return data


#### Model Run
for n_batch,(real_batch_d, real_batch_g) in enumerate(zip(test_dataloader_fine, test_dataloader_coarse)):
    # 1. Train Discriminator
    fine_data = Variable(images_to_vectors(real_batch_d, rows_test_fine, columns_test_fine))
    fine_data = (fine_data.to(torch.float32))-273.15
    generator_input = Variable(images_to_vectors(real_batch_g, rows_test_coarse, columns_test_coarse))
    generator_input = generator_input.to(torch.float32)
    fake_data = net(generator_input)
    generator_input_rescale = (rescale_data(generator_input,max_hr,min_hr))-273.15
    #### Saving downscaled Product ERA5-DownGAN
    fake_data_rescale = (rescale_data(fake_data,max_hr,min_hr))-273.15 #Celsius


#### Saving ERA5-DownGAN, a new HR dataset by cGAN
ds = xr.open_dataset(fine_resolution_test)

# lon and lat
latitudes = ds['rlat'].values  # Cambia 'latitude' se il nome della variabile è diverso
longitudes = ds['rlon'].values  # Cambia 'longitude' se il nome della variabile è diverso

# Convert fake_data_rescale to NumPy array if it's in PyTorch tensor format.
fake_data_rescale_np = fake_data_rescale.detach().cpu().numpy()  # Convert to NumPy

# Create the DataArray for xarray
fake_data_xr = xr.DataArray(
    fake_data_rescale_np, 
    dims=["batch", "latitude", "longitude"], 
    coords={"batch": range(fake_data_rescale_np.shape[0]), 
            "latitude": latitudes, 
            "longitude": longitudes}
)

# Save the downscaled data to NetCDF
fake_data_xr.to_netcdf('ERA5-DownGAN.nc', mode='w')

# Optionally print out the path or status
print("The downscaled data has been saved to 'fake_data_rescale.nc'")

