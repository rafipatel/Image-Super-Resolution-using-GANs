import os
import torch
from torch.utils import data
from PIL import Image as PILImage
import numpy as np
from torchvision import transforms

# Class to manipulate Div2K dataset
class Div2KDataset(data.Dataset):
    # Initialise dataset and variables
    def __init__(self, **kwargs):
        self.data_dir = os.path.join(kwargs['root_dir'], kwargs['source_dir'])
        self.img_size = kwargs['image_size']
        self.imgs = os.listdir(self.data_dir)
    
    # Transform image
    def transform_img(self, img):
        h_, w_ = self.img_size, self.img_size
        im_size = tuple([h_, w_])
        transform_image = transforms.Compose([
                               transforms.Resize(im_size),
                               transforms.CenterCrop(im_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        img = transform_image(img)
        return img
    
    # Load image
    def load_img(self, idx):
        im = PILImage.open(os.path.join(self.data_dir, self.imgs[idx]))
        im = self.transform_img(im)
        return im

    # Gather number of images
    def __len__(self):
        return len(self.imgs)

    # Return an image
    def __getitem__(self, idx):
        X = self.load_img(idx)
        return idx, X
