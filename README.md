# cGAN-Downscaling
cGAN-Downscaling is a model based on a Conditional Generative Adversarial Network (cGAN) structure, specifically designed for statistical climate downscaling. It produces ERA5-DownGAN dataset, high-resolution (~2.2 km) daily 2m temperature and precipitation datasets for the Italian Peninsula from ERA5 (~31 km), offering low computational costs and flexible adaptability.
## Installation

To run the cGAN-Downscaling model, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ileniam/cGAN-Downscaling.git
    cd cGAN-Downscaling
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Training scripts:
    ```bash
    python Training_ERA5-DownGAN.py
    ```

3. Run the Test scripts:
    ```bash
    python Test_ERA5-DownGAN.py
    ```
# License (None in peer review process)
This repository is private and shared exclusively for the peer review process. Please refrain from redistributing, modifying, or using the code for any other purposes until the paper is published. After publication, the code and datasets used will be made publicly available under the Apache License 2.0 to ensure the complete replicability of the experiments conducted. Additionally, the ERA5-DownGAN dataset produced will be uploaded to the CMCC Data Delivery System (DDS) and associated with a DOI.
