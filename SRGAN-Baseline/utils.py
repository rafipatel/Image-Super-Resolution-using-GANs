from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as FT
import torch
import yaml
import argparse
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def create_data_lists(train_folders, val_folders, test_folders, min_size, output_folder):
    """
    Create lists for images in the training set, validation set, and each of the test sets.

    :param train_folders: folders containing the training images; these will be merged
    :param val_folders: folders containing the validation images; these will be merged
    :param test_folders: folders containing the test images; each test folder will form its own test set
    :param min_size: minimum width and height of images to be considered
    :param output_folder: save data lists here
    """
    print("\nCreating data lists... this may take some time.\n")

    # Training data
    train_images = []
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode = "r")
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("There are %d images in the training data.\n" % len(train_images))
    with open(os.path.join(output_folder, "train_images.json"), "w") as j:
        json.dump(train_images, j)

    # Validation data
    val_images = []
    for d in val_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode = "r")
            if img.width >= min_size and img.height >= min_size:
                val_images.append(img_path)
    print("There are %d images in the validation data.\n" % len(val_images))
    with open(os.path.join(output_folder, "val_images.json"), "w") as j:
        json.dump(val_images, j)

    # Test data
    for d in test_folders:
        test_images = []
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode = "r")
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("There are %d images in the %s test data.\n" % (len(test_images), test_name))
        with open(os.path.join(output_folder, test_name + "_test_images.json"), "w") as j:
            json.dump(test_images, j)

    print("JSONs containing lists of Train, Validation, and Test images have been saved to %s\n" % output_folder)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of "pil" (PIL image), "[0, 1]" or "[-1, 1]" (pixel value ranges)
    :param target: target format, one of "pil" (PIL image), "[0, 255]", "[0, 1]", "[-1, 1]" (pixel value ranges),
                   "imagenet-norm" (pixel values standardized by imagenet mean and std.),
                   "y-channel" (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {"pil", "[0, 1]", "[-1, 1]", "imagenet-norm"}, "Cannot convert from source format %s!" % source
    assert target in {"pil", "[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm",
                      "y-channel"}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == "pil":
        img = FT.to_tensor(img)

    elif source == "[0, 1]":
        pass  # already in [0, 1]

    elif source == "[-1, 1]":
        img = (img + 1.) / 2.
    
    elif source == "imagenet-norm":
        if img.ndimension() == 3:
            img = img * imagenet_std + imagenet_mean
        elif img.ndimension() == 4:
            img = img * imagenet_std_cuda + imagenet_mean_cuda

    # Convert from [0, 1] to target
    if target == "pil":
        img = FT.to_pil_image(img)

    elif target == "[0, 255]":
        img = 255. * img

    elif target == "[0, 1]":
        pass  # already in [0, 1]

    elif target == "[-1, 1]":
        img = 2. * img - 1.

    elif target == "imagenet-norm":
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == "y-channel":
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        split: one of "train", "val", or "test"
        crop_size: crop size of HR images
        scaling_factor: LR images will be downsampled from the HR images by this factor
        lr_img_type: the target format for the LR image; see convert_image() above for available formats
        hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {"train", "val", "test"}

    def __call__(self, img):
        """
        img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        return: LR and HR images in the specified format
        """

        # Crop
        if self.split == "train":
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source="pil", target=self.lr_img_type)
        hr_img = convert_image(hr_img, source="pil", target=self.hr_img_type)

        return lr_img, hr_img

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    optimizer: optimizer with the gradients to be clipped
    grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Process settings from a YAML file.")
    parser.add_argument("--config", type = str, default = "config.yaml", help = "Path to YAML configuration file")
    return parser.parse_args()

def read_settings(config_path):
    with open(config_path, "r") as file:
        settings = yaml.safe_load(file)
    return settings

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    optimizer: optimizer whose learning rate must be shrunk.
    shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"]))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)