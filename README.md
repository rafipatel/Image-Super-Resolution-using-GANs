# Image-Super-Resolution-using-GANs



## Image Super Resolution using Generative Adversarial Networks (GANs)

This repository implements a simple Image Super Resolution (SR) model using Generative Adversarial Networks (GANs). The goal is to upscale low-resolution images to a higher resolution while preserving details and reducing artifacts.

### Getting Started

**Prerequisites:**



**Installation:**

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/Image-Super-Resolution-using-GANs.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Train the model:

   ```bash
   python train.py
   ```

   This script trains the SR-GAN model on a dataset of low-resolution and high-resolution image pairs. The script allows you to customize various parameters like the number of epochs, batch size, and model architecture.

2. Evaluate the model:

   ```bash
   python evaluate.py
   ```

   This script evaluates the trained model on a separate test set and provides metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to assess the quality of the generated super-resolution images.

3. Generate super-resolution images:

   ```bash
   python predict.py <low_resolution_image_path>
   ```

   This script takes a low-resolution image path as input and generates a super-resolution version of the image using the trained model.

### Contributing

We welcome contributions to this project! Feel free to fork the repository and submit pull requests with your improvements.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

**Note:**

* This is a basic example and can be extended in various ways. You can experiment with different GAN architectures, loss functions, and evaluation metrics.
* Include the dataset you used for training and evaluation in your repository or provide instructions on how to obtain it, if possible. 
