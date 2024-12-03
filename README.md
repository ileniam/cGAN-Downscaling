# cGAN-Downscaling
cGAN-Downscaling is a model based on a Conditional Generative Adversarial Network (cGAN) structure, specifically designed for statistical climate downscaling. It produces ERA5-DownGAN dataset, high-resolution (~2.2 km) daily 2m temperature and precipitation datasets for the Italian Peninsula from ERA5 (~31 km), offering low computational costs and flexible adaptability.
## Installation

To run the cGAN-Downscaling model, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ileniam/ERA5-DownGAN.git
    cd ERA5-DownGAN
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```
## Data
The selected dataset spans fifteen years, divided into two distinct periods: a training period (01/1990–12/2000) and a test period (01/2001–12/2005). During the training period both the low-resolution and the high-resolution dataset are used, while during the test period only the high-resolution dataset is used in the cGAN statistical downscaling model. The high-resolution dataset is also taken into account to validate the results obtained during the test period by the cGAN to judge the goodness of these at the same horizontal resolution.
#### ERA5 (~ 31km)
ERA5 reanalysis data has an horizontal resolution of 0.25 degrees (≃ 31 km; Hersbach et al., 2020), and can be accessed online at (https://doi.org/10.24381/cds.adbb2d47). 
#### VHR_REA-IT (~ 2.2km)
The very High-Resolution Dynamical Downscaling of ERA5 Reanalysis (VHR_REA- IT) is chosen as the high-resolution dataset (Fig. 1b). This reanalysis is at the convection-permitting scale (horizontal grid spacing 0.02°, ≃2.2 km; Raffa et al., 2021) by COSMO in CLimate Mode (COSMO-CLM) on a domain covering the Italian Peninsula, described by Raffa, M. et al. (2021) available for download at https://doi.org/10.25424/cmcc/era5-2km_italy. 
## Usage
#### Pre-processing
Run the Normalization script:
```bash
python Normalization_ERA5-DownGAN.py
```
#### Training
Run the Training script:
```bash
python Training_ERA5-DownGAN.py
```
#### Test
Run the Test script:
```bash
python Test_ERA5-DownGAN.py
```

## Results
During the training phase, the model optimizes both the generator and discriminator networks. The optimal generator, learned through training, is then used to generate high-resolution downscaled data from ERA5 during the testing phase. Specifically, the trained generator applies to the ERA5 dataset from the test phase to produce a new downscaled dataset at 2.2 km resolution. 
## Example of validation comparing dynamical (VHR_REA-IT) and statistical downscaling based cGAN (ERA5-DownGAN)
#### Some Figures
![Fig_6](https://github.com/user-attachments/assets/01641900-9aa4-4931-bf9b-aa900ce3963f)
## License (None in peer review process)

This repository is private and shared exclusively for the peer review process. Please refrain from redistributing, modifying, or using the code for any other purposes until the paper is published. After publication, the code and datasets used will be made publicly available under the Apache License 2.0 to ensure the complete replicability of the experiments conducted. Additionally, the ERA5-DownGAN dataset produced will be uploaded to the CMCC Data Delivery System (DDS) and associated with a DOI.
