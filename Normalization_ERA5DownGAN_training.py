#!/usr/bin/env python
# coding: utf-8

# ## cGAN Downscaling
# Ilenia Manco

#### Imports

import os
import pandas as pd
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


### INPUT

target_path = xr.open_dataset('./data/input/ERA5_1990-2000_originale_interp.nc')
lr_path = xr.open_dataset('./data/input/ERA5_1990-2000_originale.nc')
hr_path = xr.open_dataset('./data/input/VHR_1990-2000_originale.nc')

#### Target min-max

max_target = target_path.t2m.max()
min_target = target_path.t2m.min()
min_target = float(min_target)
max_target = float(max_target)


#### HR min-max

max_hr = hr_path.t2m.max()
min_hr = hr_path.t2m.min()
max_hr = float(max_hr)
min_hr = float(min_hr)

#### LR min-max

max_lr = lr_path.t2m.max()
min_lr = lr_path.t2m.min()
max_lr = float(max_lr)
min_lr = float(min_lr)

#### Normalization Function
def scale_data (data):
     data = ( (data - min_value)/(max_value-min_value))*(1-(-1))+(-1)
     return data


#### TARGET Normalization

target_scale = scale_data(target_path['t2m'])
max_target_scale = target_scale.max()
min_target_scale = target_scale.min()
print(min_target_scale,max_target_scale)

target_scale_hr = scale_data(target_path['t2m'])
max_target_scale_hr = target_scale_hr.max()
min_target_scale_hr = target_scale_hr.min()
print(min_target_scale_hr,max_target_scale_hr)

#### LR Normalization
lr_scale_target = scale_data(lr_path['t2m'])
max_lr_scale_target = lr_scale_target.max()
min_lr_scale_target = lr_scale_target.min()
print(min_lr_scale_target,max_lr_scale_target)

lr_scale_hr = scale_data(lr_path['t2m'])
max_lr_scale_hr = lr_scale_hr.max()
min_lr_scale_hr = lr_scale_hr.min()
print(min_lr_scale_hr,max_lr_scale_hr)

lr_scale_lr = scale_data(lr_path['t2m'])
max_lr_scale_lr = lr_scale_lr.max()
min_lr_scale_lr = lr_scale_lr.min()
print(min_lr_scale_lr,max_lr_scale_lr)

#### HR Normalization
hr_scale_target = scale_data(hr_path['t2m'])
max_hr_scale_target = hr_scale_target.max()
min_hr_scale_target = hr_scale_target.min()
print(min_hr_scale_target,max_hr_scale_target)

hr_scale_hr = scale_data(hr_path['t2m'])
max_hr_scale_hr = hr_scale_hr.max()
min_hr_scale_hr = hr_scale_hr.min()
print(min_hr_scale_hr,max_hr_scale_hr)


#### Saving NetCDF

# TARGET
target_scale.to_netcdf('./data/training_dataset/normalization_target/target_scale.nc', mode='w', format=None)
target_scale_hr.to_netcdf('./data/training_dataset/normalization_vhr/target_scale_hr.nc', mode='w', format=None)

# HR
hr_scale_target.to_netcdf('./data/training_dataset/normalization_target/hr_scale_target.nc', mode='w', format=None)
hr_scale_hr.to_netcdf('./data/training_dataset/normalization_vhr/hr_scale_hr.nc', mode='w', format=None)

# LR
lr_scale_target.to_netcdf('./data/training_dataset/normalization_target/lr_scale_target.nc', mode='w', format=None)
lr_scale_hr.to_netcdf('./data/training_dataset/normalization_vhr/lr_scale_hr.nc', mode='w', format=None)
lr_scale_lr.to_netcdf('./data/training_dataset/normalization_era5/lr_scale_lr.nc', mode='w', format=None)

