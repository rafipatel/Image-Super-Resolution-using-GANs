from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import torch
import numpy as np
from models import SRResNet, Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "./"
test_data_names = ["Set5", "Set14", "BSDS100", "DIV2K_valid_HR"]

# Model checkpoints
srgan_checkpoint = "checkpoint_srgan.pth.tar"
srresnet_checkpoint = "checkpoint_srresnet.pth.tar"

#### Note ####
# Many updates were made to the model classes and their functionality, gradually adding the use of different optimisers, self-attention, etc.
# Since we saved the whole model without exporting it as dictionaries, we need to call the respective model class (SRResNet/Generator) with its relevant parameters
# The parameters are then updated for the model as seen below
# Some of our models will allow you to skip the steps uncommented below, here is how you can do so - this only works for the most up to date models we ran, which are those with attention
# Load SRResNet
#srresnet = torch.load(srresnet_checkpoint)["model"].to(device)
#srresnet.eval()
#model = srresnet
# Load SRGAN
#srgan_generator = torch.load(srgan_checkpoint)["generator"].to(device)
#srgan_generator.eval()
#model = srgan_generator
########################################################################################################################

# Load SRResNet
net = SRResNet(large_kernel_size = 9, small_kernel_size = 5, n_channels = 64, n_blocks = 20, scaling_factor = 4, self_attention = False)
srresnet = torch.load(srresnet_checkpoint)["model"].to(device)
net.load_state_dict(srresnet.state_dict())
model = net.to(device)
model.eval()

# Load SRGAN (uncomment and comment SRResNet lines to use this)
#net = Generator(large_kernel_size = 9, small_kernel_size = 3, n_channels = 64, n_blocks = 16, scaling_factor = 4, self_attention = False)
#srgan_generator = torch.load(srgan_checkpoint)["generator"].to(device)
#net.load_state_dict(srgan_generator.state_dict())
#model = net.to(device)
#model.eval()

# Evaluate
def evaluation():
    for test_data_name in test_data_names:
        print(f"\nFor {test_data_name}:\n")

        # Custom dataloader
        test_dataset = SRDataset(data_folder, split = "test", crop_size = 0, scaling_factor = 4, lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]", test_data_name = test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = True)

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = []
        SSIMs = []

        # Faster computation
        with torch.no_grad():
            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # Calculate PSNR and SSIM
                sr_imgs_y = convert_image(sr_imgs, source = "[-1, 1]", target = "y-channel").squeeze(0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source = "[-1, 1]", target = "y-channel").squeeze(0)  # (w, h), in y-channel
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range = 255.0)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range = 255.0)
                PSNRs.append(psnr)
                SSIMs.append(ssim)

        # Print average PSNR and SSIM
        print(f"Number of images: {len(test_loader)}")
        print(f"Average PSNR: {round(np.mean(PSNRs), 4)}")
        print(f"Average SSIM: {round(np.mean(SSIMs), 4)}")

        print("\n")

if __name__ == "__main__":
    evaluation()