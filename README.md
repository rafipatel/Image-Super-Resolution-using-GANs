# Super-Resolution with SRResNet and SRGAN

Note: This repository and its contents support the coursework of the INM705 module at City, University of London.

## Super-Resolution Generative Adversarial Network Original Architecture
![Example Image](architecture.jpg)

You can find the original architecture of SRGAN in the following paper: [Photo-Realistic Single Image Super-Resolution Using Generative Adversarial Networks](https://arxiv.org/abs/1609.04802).

We recommend reading the paper to understand the problem of image super-resolution using the Super-Resolution Residual Network and Super-Resolution Generative Adversarial Network models.

### Overview

This repository contains a PyTorch implementation of the Super-Resolution Residual Network (SRResNet) and Super-Resolution Generative Adversarial Network (SRGAN) models for enhancing the resolution of images. SRResNet and SRGAN are deep learning architectures capable of generating high-resolution images from low-resolution inputs.

### Introduction
Super-resolution is a computer vision task that aims to improve the resolution of an image. These models utilizes a deep neural network architecture to enhance the details and quality of low-resolution images. This implementation includes training and evaluation scripts, along with utilities for data loading, logging, and model checkpoints.

**Super-Resolution Residual Network (SRResNet)**

SRResNet is a deep residual network. The SRResNet architecture is inspired by the ResNet architecture and is optimized for super-resolution tasks. It learns to map low-resolution images to high-resolution ones by capturing intricate details and features through multiple layers.

**Super-Resolution Generative Adversarial Network (SRGAN)**

SRGAN combines the SRResNet with an adversarial network, consisting of a discriminator and a generator. The discriminator learns to differentiate between real high-resolution images and generated high-resolution images, while the generator aims to produce high-quality images that are indistinguishable from real ones. This adversarial training process encourages the generator to generate more realistic and visually pleasing high-resolution images.

### Features
- Implementation of SRGAN architecture with both generator and discriminator networks.
- Implementation of SRResNet architectures.
- Supports different loss functions for content and adversarial losses.
- Training pipeline with logging using Weights & Biases ('wandb').
- Checkpointing mechanism to save and resume training from a specific epoch.

### Requirements
- Python 3.9+
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/)
- All other requirements listed in [**requirements.txt**](https://github.com/rafipatel/Image-Super-Resolution-using-GANs/blob/main/requirements.txt)

### Training
The training process involves optimising the SRGAN and SRResNet models to generate high-quality images. You can customise various hyperparameters such as batch size, learning rate, optimiser, and loss function in the `config.yaml` file. Please refer to the `config.yaml` file for a full list of hyperparameters.

### File Structure ('/SRGAN-Baseline' and '/SRGAN-Final')
- `train_srresnet.py`: Script used to train an SRResNet model with the 'SRResNet' hyperparameters set in the `config.yaml` file.
- `train_srgan.py`: Script used to train an SRGAN model with the 'SRGAN' hyperparameters set in the `config.yaml` file. A checkpointed/pre-trained SRResNet model can be used as the generator, rather than training from scratch (recommended).
- `config.yaml`: File to edit hyperparameters, including those for SRResNet and SRGAN. Additional hyperparameters are included for choosing project and logger names for 'wandb'. A set of 'Data' parameters are included for finding the folder containing the lists of images generated by the `create_data_lists.py` file.
- `create_data_lists.py`: Used to create lists of images in the training set, validation set, and each of the test sets. The images in these lists satisfy a minimum image size (min_size).
- `models.py`: Contains the definitions of the generator and discriminator networks.
- `datasets.py`: Defines custom datasets and data loaders.
- `utils.py`: Utility functions for image conversion, etc.
- `logger.py`: Logger class for logging training metrics to 'wandb'.
- `loss.py`: Custom VGG loss function used in training.
- `eval.py`: Used to calculate the peak signal-to-noise ratio, structural similarity index, and if an SRGAN model, the Fréchet inception distance for the test sets (or any other set of images).
- `checkpoints/`: Directory to save model checkpoints during training, and can be used for inference.
- `train_images.json`: List of images for the training set, generated by running `create_data_lists.py`. These images are used during the training phase, i.e. by `train_srresnet.py` or `train_srgan.py`
- `val_images.json`: List of images for the validation set, generated by running `create_data_lists.py`. These images are used during the validation phase, i.e. by running `train_srresnet.py`. There is no validation phase while training GANs.
- `{test_set_name}_test.json`: List of images for the test set(s), generated by running `create_data_lists.py`. These images are used during the testing phase, i.e. by running `eval.py`.
- `inference.ipynb`: Jupyter Notebook used for inference. Allows for an image to be input and to examine the super-resolved outputs of the SRResNet and SRGAN checkpoints.
- `super_resolve.py`: File for inference, a simple alternative to the Jupyter Notebook. Creates a figure containing a comparison of a checkpointed SRResNet and a checkpointed SRGAN, alongside the original high-resolution image and a bicubic interpolated image.

### Datasets
Our training dataset ('test2015') and validation dataset ('val2017') can be downloaded [here](https://cocodataset.org/#download). If you choose to train/validate using different datasets, you must edit the `create_data_lists.py` file parameters with the directories to these datasets. The code as it is assumes that you use the same datasets as us (and `create_data_lists.py` does not need to be run as we already include the '.json' files for you, only if using our datasets).

Our test datasets ('Set5', 'Set14', and 'BSDS100') can be downloaded [here](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u). If you choose to test using different datasets, you must edit the `create_data_lists.py` file parameters with the directories to these datasets. The code as it is assumes that you use the same datasets as us (and `create_data_lists.py` does not need to be run as we already include the '.json' files for you, only if using our datasets). You need to edit the data hyperparameters at the beginning of the `eval.py` file with the names of these dataset folders, and the location of them (only if you are not using our datasets).

### Checkpoints
Our checkpoints can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EvPZOvxznetMi6MV3iN40JsBosC_QSkUhjvD464jKtUYrg?e=barQWp). Each folder contains a 'readme.txt' file containing the hyperparameters used to train the models. You will need these if you edit the code later, otherwise we have already filled them in appropriately for you. You only need to replace the dummy checkpoints in this repository with those found [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EvPZOvxznetMi6MV3iN40JsBosC_QSkUhjvD464jKtUYrg?e=barQWp).

Unfortunately, our SRResNet checkpoints were made before we introduced residual scaling in the code. As a result, there are no residual scaling attributes in these checkpoints, meaning that continuing training from these checkpoints is not possible. It is possible to avoid this by removing parts of the code that reference 'self.resid_scale_factor', though we do not do this here. Inference is still possible regardless. The SRGAN checkpoints will run as normal.

### Usage (detailed)
1. We recommend first downloading/cloning the whole repository (bar the `/archive` folder), though if you wish to work only with the baseline model you do not need the `/SRGAN-Final` folder, and vice versa for working with our final model. We also recommend sticking to the folder structure found in this repository, otherwise you will need to make a few edits indicating where the datasets and checkpoints can be found.

2. Secondly, you should ensure that all the libraries listed in the [**requirements.txt**](https://github.com/rafipatel/Image-Super-Resolution-using-GANs/blob/main/requirements.txt) file have been installed on your environment (you may wish to create a new environment from scratch to avoid dependency conflicts). You can use your favourite method to install these libraries, such as through using the `pip install -r requirements.txt` command in your terminal. You must also install [PyTorch](https://pytorch.org/get-started/locally/), ideally the CUDA version if you wish to work with a GPU. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation.

3. Now, you will need to download a training set, validation set, and one or more test sets. In our work, we used 'test2015' from MSCOCO as our training set, 'val2017' from MSCOCO as our validation set, and our test sets are the standard sets for image super-resolution, namely 'Set5', 'Set14', and 'BSDS100'. We have created folders for these already in our repository, and the download links for each of these datasets can be found inside the respective folders. We recommend deleting our folders and replacing them with the folders that you download. In other words, download the datasets, delete our dummy dataset folders, and extract the dataset contents (which should be in a folder with the same name as our dummy folders) and put them in the same location as where the dummy dataset folders were (see repository, you essentially need the dataset folders outside of the `/SRGAN-Baseline` and `/SRGAN-Final` folders).

4. **IMPORTANT IF USING DIFFERENT DATASETS**: After downloading the datasets, before you can start running any code, you may need to run the `create_data_lists.py` file inside the `/SRGAN-Baseline` and `/SRGAN-Final` folders. If you are not using the same datasets as us, you will certainly need to edit the contents of this file and later run it. In particular, you need to edit the following parameters with the directories of your datasets:

``train_folders = ["../test2015"]
val_folders = ["../val2017"]
test_folders = ["../Set5", "../Set14", "../BSDS100"]``

The 'test_folders' is only used during testing, and it is not mandatory to include a dataset here if you are only interested in model training. We note that if you follow our guidance and are using the same     datasets as us, you can ignore this step. We have already included the 'train_images.json' and 'val_images.json' files for you, which essentially inform the program which images are at least of a certain        minimum size (we chose 100) and can be safely used during training.

5. After completing all of the previous steps, you can safely run the `train_srresnet.py` file from your chosen folder (`/SRGAN-Baseline` or `/SRGAN-Final`) if you wish to train an SRResNet model, or you can run the `train_srgan.py` file if you wish to train an SRGAN model. You may edit the `config.yaml` file with your chosen hyperparameters to use during training, including adding any checkpoint paths (such as those that we provide). We advise using an SRResNet checkpoint as the generator for the SRGAN when running the 'train_srgan.py' file, though it is not compulsory.

6. **Optional**: Should you wish to use our checkpoints, you need to download them [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EvPZOvxznetMi6MV3iN40JsBosC_QSkUhjvD464jKtUYrg?e=barQWp) for your choice of model. You need to edit the 'config.yaml' file with the checkpoint path if you interested in using our checkpoints for the relevant model. These checkpoints should be placed in the `/checkpoints` folder for the relevant model folder.

7. **IMPORTANT NOTE**: The code assumes that you open your workspace in either the `/SRGAN-Baseline` or `/SRGAN-Final` model folders in your integrated development environment (IDE), rather than opening the whole folder containing all of the datasets, etc. You can edit these to your preferences, but we recommend following the folder structure set out in this repository for ease.

### How to Run (Simple Words)
1. Clone the repo
2. Change directory to desired model (SRGAN-Baseline or SRGAN-Final)
2. Download images for Train, Valid, Test dataset folders. (For instance for tryout, You can download random 20 images, split them in 3 different folders) 
3. Update the folder name in (`create_data_lists.py`)
4. Run (`create_data_lists.py`)
```
python create_data_lists.py
```
5. Run (`tran_srresnet.py`) or (`train_srgan.py`) for training the model.


### Inference
We provide a Jupyter Notebook (`inference.ipynb`) and standard Python file (`super_resolve.py`) for inference. Either of these methods can be used to super-resolve an image. You should enter a path to a high-resolution image in the 'visualise_sr()' function in your chosen inference file. We provide example paths already if you are using our test datasets. You must have our checkpoints downloaded and added to the `checkpoints/` folder, though you can edit the paths as you wish to the checkpoints inside the code. We recommend following the Jupyter Notebook for an easy to understand demonstration.

### Weights & Biases ('wandb')
If you have not used 'wandb' previously, you will be prompted to enter your API key into the terminal. You need a (free) 'wandb' account if not already made, and you can find further instructions [here](https://docs.wandb.ai/quickstart).

## Acknowledgements
- The SRResNet model architecture and training procedure are based on the paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
- This implementation borrows concepts from various open-source repositories and research papers in the field of image super-resolution
- Code structure and design is mostly based on the [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/) repository, with our own major additions

Feel free to contribute, report issues, or suggest improvements to this repository. Happy coding!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
