import torch
from PIL import Image
import albumentations as A #for data augmentation
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
# CHECKPOINT_GEN = "gen.pth.tar" # CHECKPOINT FILES
# CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 16
NUM_WORKERS = 0
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3
OPTIMIZER = "adam"

#After both_transform, for High res image will normalize it between -1 and 1, then convert it into tensor
highres_transform = A.Compose(
    [
        A.Normalize(mean=[-1, -1, -1], std=[1, 1, 1]), # as per paper
        ToTensorV2(),
    ]
)

# After both_transform, for low resol, take that high quality which is outputted from both_transforms
# make it to 24x24 low resolution, then normalize it b/w 0 and 1 as per paper
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(), #converts to pytorch format (C,H,W)
    ]
)

# Get the image, random crop (eg :-96,96)
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(), #converts to pytorch format (C,H,W)
    ]
)