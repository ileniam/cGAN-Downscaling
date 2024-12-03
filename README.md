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
## Usage
### Pre-processing
Run the Normalization script:
```
bash
python Normalization_ERA5-DownGAN.py
```
#### Training
Run the Training script:
```
bash
python Training_ERA5-DownGAN.py
```
#### Test
Run the Test script:
```
bash
python Test_ERA5-DownGAN.py
```

## Results
During the training phase, the model optimizes both the generator and discriminator networks. The optimal generator, learned through training, is then used to generate high-resolution downscaled data from ERA5 during the testing phase. Specifically, the trained generator applies to the ERA5 dataset from the test phase to produce a new downscaled dataset at 2.2 km resolution. 

## License (None in peer review process)
This repository is private and shared exclusively for the peer review process. Please refrain from redistributing, modifying, or using the code for any other purposes until the paper is published. After publication, the code and datasets used will be made publicly available under the Apache License 2.0 to ensure the complete replicability of the experiments conducted. Additionally, the ERA5-DownGAN dataset produced will be uploaded to the CMCC Data Delivery System (DDS) and associated with a DOI.
