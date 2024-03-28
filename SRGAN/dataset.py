import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import torch
from torch.utils import data
from PIL import Image as PILImage
import numpy as np
from torchvision import transforms
import config

# Allows for multiple datasets in different folders, e.g. ["DIV2K_train_HR", "Flickr2K"]
class SuperResolutionDataset(Dataset):
    def __init__(self, dataset_folders, root_dir = os.getcwd()):
        super(SuperResolutionDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir

        #os.path.join(root_dir, 'image_pairs')  # Assuming image_pairs subfolder
        # hr_img_dir = self.root_dir
        # hr_img_dir = os.path.join(self.root_dir, "hr_img")

        for dataset_folder in dataset_folders:
            dataset_dir = os.path.join(root_dir, dataset_folder)
            for hr_img in sorted(os.listdir(dataset_dir)):
                hr_img_path = os.path.join(dataset_dir, hr_img)
                self.data.append(hr_img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = np.array(Image.open(img_path))

        transformed_img = config.both_transforms(image = img)['image']
        lowres_image = config.lowres_transform(image = transformed_img)['image']
        highres_image = config.highres_transform(image = transformed_img)['image']
        return lowres_image, highres_image

# data = SuperResolutionDataset(root_dir="G:\\My Drive\\GAN\\")
