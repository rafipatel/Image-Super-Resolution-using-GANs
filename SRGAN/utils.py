import torch
import os
# import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import config

# def gradient_penalty(critic, real, fake, device): #actually not needed
#     BATCH_SIZE, C, H, W = real.shape
#     alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
#     interpolated_images = real * alpha + fake.detach() * (1 - alpha)
#     interpolated_images.requires_grad_(True)

#     # Calculate critic scores
#     mixed_scores = critic(interpolated_images)

#     # Take the gradient of the scores with respect to the images
#     gradient = torch.autograd.grad(
#         inputs=interpolated_images,
#         outputs=mixed_scores,
#         grad_outputs=torch.ones_like(mixed_scores),
#         create_graph=True,
#         retain_graph=True,
#     )[0]
#     gradient = gradient.view(gradient.shape[0], -1)
#     gradient_norm = gradient.norm(2, dim=1)
#     gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
#     return gradient_penalty


# def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     checkpoint = {
#         "state_dict": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(checkpoint, filename)


# def load_checkpoint(checkpoint_file, model, optimizer, lr):
#     print("=> Loading checkpoint")
#     checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])

#     # If we don't do this then it will just have learning rate of old checkpoint
#     # and it will lead to many hours of debugging \:
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


# Here, will send a folder of low res images, and run this plot_examples,it will upscale the image with current generator to get high upscaled images
def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:

        image = "/root/tensorflow_datasets/downloads/extracted/ZIP.data.visi.ee.ethz.ch_cvl_DIV2_DIV2_vali_LRpQpdHEuI3k6NMA2PsrkExS_pOyspikiZaXdg18u21VM.zip/DIV2K_valid_LR_bicubic/X4/0801x4.png"
        image = Image.open(image)
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, file)
    gen.train()