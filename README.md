# Super-Resolution with SRResNet and SRGAN

Note: This repository and its contents support the coursework of the INM705 module at City, University of London.

### Overview

This repository contains a PyTorch implementation of the Super-Resolution Generative Adversarial Network (SRGAN) & Super-Resolution Residual Network (SRResNet) for enhancing the resolution of images. SRGAN and SRResNet are deep learning architectures capable of generating high-resolution images from low-resolution inputs.

### Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Features](#features)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [File Structure](#FileStructure)
7. [Usage](#usage)

### Introduction
Super-resolution is a computer vision task that aims to improve the resolution of an image. These models utilizes a deep neural network architecture to enhance the details and quality of low-resolution images. This implementation includes training and evaluation scripts, along with utilities for data loading, logging, and model checkpoints.

**Super-Resolution Residual Network (SRResNet)**

SRResNet is a deep residual network. The SRResNet architecture is inspired by the ResNet architecture and is optimized for super-resolution tasks. It learns to map low-resolution images to high-resolution ones by capturing intricate details and features through multiple layers.

**Super-Resolution Generative Adversarial Network (SRGAN)**

SRGAN combines the SRResNet with an adversarial network, consisting of a discriminator and a generator. The discriminator learns to differentiate between real high-resolution images and generated high-resolution images, while the generator aims to produce high-quality images that are indistinguishable from real ones. This adversarial training process encourages the generator to generate more realistic and visually pleasing high-resolution images.

### Requirements
- Python 3.9+
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/)
- wandb (Weights & Biases)
- All other requirements listed in [**requirements.txt**](https://github.com/rafipatel/Image-Super-Resolution-using-GANs/blob/main/requirements.txt)

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
- `loss.py`: Custom VGG loss function used in training.
- `checkpoints/`: Directory to save model checkpoints during training.

### Usage
1. We recommend first downloading/cloning the whole repository (bar the '/archive' folder), though if you wish to work only with the baseline model you do not need the '/SRGAN-Final' folder, and vice versa for working with our final model. We also recommend sticking to the folder structure found in this repository.

2. Secondly, you should ensure that all the libraries listed in the [**requirements.txt**](https://github.com/rafipatel/Image-Super-Resolution-using-GANs/blob/main/requirements.txt) file have been installed on your environment (you may wish to create a new environment from scratch to avoid dependency conflicts). You must also install [PyTorch](https://pytorch.org/get-started/locally/), ideally the CUDA version if you wish to work with a GPU. 

3. Now, you will need to download a training set, validation set, and one or more test sets. In our work, we used 'test2015' from MSCOCO as our training set, 'val2017' from MSCOCO as our validation set, and our test sets are the standard sets for image super-resolution, namely 'Set5', 'Set14', and 'BSDS100'. We have created folders for these already in our repository, and the download links for each of these datasets can be found inside the respective folders. We recommend deleting our folders and replacing them with the folders that you download. In other words, download the datasets, delete our dummy dataset folders, and extract the dataset contents (which should be in a folder with the same name as our dummy folders) and put them in the same location as where the dummy dataset folders were (see repository, you essentially need the dataset folders outside of the 'SRGAN' folders).

4. **IMPORTANT IF USING DIFFERENT DATASETS**: After downloading the datasets, before you can start running any code, you may need to run the 'create_data_lists.py' file inside the '/SRGAN-Baseline' and '/SRGAN-Final' folders. If you are not using the same datasets as us, you will certainly need to edit the contents of this file and later run it. In particular, you need to edit the following parameters with the directories of your datasets:

  train_folders = ["../test2015"]
  val_folders = ["../val2017"]
  test_folders = ["../Set5", "../Set14", "../BSDS100"]

  The 'test_folders' is only used during testing, and it is not mandatory to include a dataset here if you are only interested in model training. We note that if you follow our guidance and are using the same     datasets as us, you can ignore this step. We have already included the 'train_images.json' and 'val_images.json' files for you, which essentially inform the program which images are at least of a certain        minimum size (we chose 100) and can be safely used during training.

5. After completing all of the previous steps, you can safely run the 'train_srresnet.py' file from your chosen folder ('/SRGAN-Baseline' or '/SRGAN-Final') if you wish to train an SRResNet model, or you can run the 'train_srgan.py' file if you wish to train an SRGAN model. You may edit the 'config.yaml' file with your chosen hyperparameters to use during training, including adding any checkpoint paths (such as those that we provide). We advise using an SRResNet checkpoint as the generator for the SRGAN when running the 'train_srgan.py' file, though it is not compulsory.

6. **Optional**: Should you wish to use our checkpoints, you need to download them [here]() for your choice of model. You do not need to make any edits to the 'config.yaml' if you are only interested in using our checkpoints for the relevant model. These checkpoints should be placed in the '/checkpoints' folder for the relevant model.

## Acknowledgements
- The SRResNet model architecture and training procedure are based on the paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
- This implementation borrows concepts from various open-source repositories and research papers in the field of image super-resolution
- Code structure and design is mostly based on the [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/) repository, with our own major additions

Feel free to contribute, report issues, or suggest improvements to this repository. Happy coding!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
