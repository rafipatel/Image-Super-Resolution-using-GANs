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

root_dir = "/content/drive/MyDrive/City, University of London /BRP/GAN/" #/ZIP.data.visi.ee.ethz.ch_cvl_DIV2_DIV2_trai_HRnQucRj9uNbbGLMapMO4iJPZws0wd-EGldnD5rPD2wzU.zip/DIV2K_train_HR/"
class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir):
        super(SuperResolutionDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir  #os.path.join(root_dir, 'image_pairs')  # Assuming image_pairs subfolder

        hr_img_dir = os.path.join(self.root_dir,"hr_img/")

        for hr_img in sorted(os.listdir(hr_img_dir)):
          hr_img_path = os.path.join(hr_img_dir,hr_img)
          self.data.append(hr_img_path)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_path = self.data[index]

        img = np.array(Image.open(img_path))

        transformed_img = both_transforms(image=img)['image']
        lowres_image = lowres_transform(image=transformed_img)['image']
        highres_image = highres_transform(image=transformed_img)['image']

        return lowres_image, highres_image
