**Super-Resolution with SRGAN and SRResNet**


### Overview

This repository contains a PyTorch implementation of the Super-Resolution Generative Adversarial Network (SRGAN) & Super-Resolution Convolutional Neural Network (SRResNet) for enhancing the resolution of images. SRGAN and SRResNet are deep learning architecture capable of generating high-resolution images from low-resolution inputs.

### Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Features](#features)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [ File Structure](#FileStructure)
7. [Usage](#usage)

### Introduction
Super-resolution is a computer vision task that aims to improve the resolution of an image. These models utilizes a deep neural network architecture to enhance the details and quality of low-resolution images. This implementation includes training and evaluation scripts, along with utilities for data loading, logging, and model checkpoints.

**SRResNet (Super-Resolution Residual Network)**

SRResNet is a deep residual network. The SRResNet architecture is inspired by the ResNet architecture and is optimized for super-resolution tasks. It learns to map low-resolution images to high-resolution ones by capturing intricate details and features through multiple layers.

**SRGAN ( Super-Resolution Generative Adversarial Network)**

SRGAN combines the SRResNet with an adversarial network, consisting of a discriminator and a generator. The discriminator learns to differentiate between real high-resolution images and generated high-resolution images, while the generator aims to produce high-quality images that are indistinguishable from real ones. This adversarial training process encourages the generator to generate more realistic and visually pleasing high-resolution images.

### Requirements
- Python 3.6+
- PyTorch
- wandb (Weights & Biases)
- All other requirements listed in **requirements.txt**

### Features
- Implementation of SRGAN architecture with both generator and discriminator networks.
- Implementation of SRResNet architectures.
- Supports different loss functions for content and adversarial losses.
- Training pipeline with logging using Weights & Biases (wandb).
- Checkpointing mechanism to save and resume training from a specific epoch.

### Training
The training process involves optimizing the SRGAN and SRResNet model to generate high-quality images. You can customize various parameters such as batch size, learning rate, optimizer, and loss function in the configuration file.

### Evaluation
After training, you can evaluate the trained model on a separate validation dataset. The evaluation script computes metrics such as loss, PSNR (Peak Signal-to-Noise Ratio), and SSIM (Structural Similarity Index) to assess the performance of the model.

### File Structure (respectively)
- `train.py`: Main script to initiate training.
- `models.py`: Contains the definitions of the generator and discriminator networks (i.e. Model Architectures).
- `datasets.py`: Defines custom datasets and data loaders.
- `utils.py`: Utility functions for image conversion, etc.
- `logger.py`: Logger class for logging training metrics to wandb.
- `loss.py`: Custom loss functions used in training.
- `checkpoints/`: Directory to save model checkpoints during training.


### Usage
1. Clone the repository:
   ```
   git clone https://github.com/rafipatel/Image-Super-Resolution-using-GANs.git
   ```

2. Change directory as required.

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your dataset by placing high-resolution images in a folder. Update the dataset path in the configuration file (`config.yaml`).

5. Train the model:
   ```
   python train.py --config config.yaml
   ```


## Acknowledgments
- The SRResNet model architecture and training procedure are based on the paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by Christian Ledig et al.
- This implementation borrows concepts from various open-source repositories and research papers in the field of image super-resolution.
- Code structure and design influenced by various open-source implementations of GANs, SRGANs and SRResNets.

Feel free to contribute, report issues, or suggest improvements to this repository. Happy coding!


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
