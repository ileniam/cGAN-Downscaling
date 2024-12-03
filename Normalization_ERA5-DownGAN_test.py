#!/usr/bin/env python
# coding: utf-8

#### cGAN Downscaling

#### Imports

import os
import pandas as pd
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#### INPUT
target_path = xr.open_dataset('./data/input/ERA5_2001-2005_originale_interp.nc') 
lr_path = xr.open_dataset('./data/input/ERA5_2001-2005_originale.nc')
hr_path = xr.open_dataset('./data/input/VHR_2001-2005_originale.nc')

#### TARGET min-max
max_target = target_path.t2m.max()
min_target = target_path.t2m.min()
min_target = float(min_target)
max_target = float(max_target)

#### HR min-max
max_hr = hr_path.t2m.max()
min_hr = hr_path.t2m.min()
min_hr = float(min_hr)
max_hr = float(max_hr)

#### LR min-max
max_lr = lr_path.t2m.max()
min_lr = lr_path.t2m.min()
min_lr = float(min_lr)
max_lr = float(max_lr)

#### Normalization Function
def scale_data (data):
     data = ( (data - min_value)/(max_value-min_value))*(1-(-1))+(-1)
     return data

#### TARGET Normalization
target_scale = scale_data(target_path['t2m'])
max_target_scale = target_scale.max()
min_target_scale = target_scale.min()
print(min_target_scale,max_target_scale)

#### LR Normalization
lr_scale = scale_data(lr_path['t2m'])
max_lr_scale = lr_scale.max()
min_lr_scale = lr_scale.min()
print(min_lr_scale,max_lr_scale)

#### HR Normalization
hr_scale = scale_data(hr_path['t2m'])
max_hr_scale = hr_scale.max()
min_hr_scale = hr_scale.min()
print(min_hr_scale,max_hr_scale)


#### Saving NetCDF

# HR
 
hr_scale.to_netcdf('./data/test_dataset/normalization_vhr/hr_scale.nc', mode='w', format=None)

# LR
lr_scale.to_netcdf('./data/test_dataset/normalization_era5/lr_scale.nc', mode='w', format=None)

# TARGET
target_scale.to_netcdf('./data/test_dataset/normalization_vhr/target_scale.nc',mode = 'w', format=None)


