# cGAN-Downscaling
cGAN-Downscaling is a model based on a Conditional Generative Adversarial Network (cGAN) structure, specifically designed for statistical climate downscaling. It produces ERA5-DownGAN dataset, high-resolution (~2.2 km) daily 2m temperature for the Italian Peninsula from ERA5 (~31 km), offering low computational costs and flexible adaptability.
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

 #### Random day from the test period of 2m-temperature (a, ERA5; b, VHR_REA-IT; c, ERA5-DownGAN)
 ![Immagine1](https://github.com/user-attachments/assets/cbb92fe0-92c3-4ab4-9c7d-69ea25eed2f8)
 
#### Error Metrics of 2m-temperature(a, BIAS; b, MAE; c, RMSE; d, correlation)
 ![Fig_6](https://github.com/user-attachments/assets/01641900-9aa4-4931-bf9b-aa900ce3963f)
 
#### PDFs of 2m-tmperature (ERA5 in green; VHR_REA-IT in blue, ERA5-DownGAN in orange)
 ![Fig_11](https://github.com/user-attachments/assets/2e3e54bb-a4e3-44b1-bcd0-3e90175ef066)

## License (Usage allowed upon citation)
This repository is publicly available. However, the use of the code and the associated dataset is permitted only upon proper citation of the methodological paper:

Ilenia Manco, Walter Riviera, Andrea Zanetti, Marco Briscolini, Paola Mercogliano, Antonio Navarra,
A new conditional generative adversarial neural network approach for statistical downscaling of the ERA5 reanalysis over the Italian Peninsula,
Environmental Modelling & Software,
Volume 188,
2025,
106427,
ISSN 1364-8152,
https://doi.org/10.1016/j.envsoft.2025.106427.

The code is released under the Apache License 2.0. The ERA5-DownGAN dataset will be made available through the CMCC Data Delivery System (DDS) and linked to a DOI for full reproducibility.

Please refrain from redistributing, modifying, or using the materials without citing the publication above.


