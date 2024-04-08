import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import piq

from models import Generator, Discriminator
from datasets import SRDataset
from utils import *
from logger import Logger
from loss import TruncatedVGG19

# Data parameters
data_folder = "./"  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "checkpoints/checkpoint_srresnet.pth.tar"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
starting_epoch = 0  # start at this epoch
iterations = 2000000  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 0.001  # the coefficient to weight the adversarial loss in the perceptual loss
learning_rate = 0.0001  # learning rate
grad_clip = 0.1  # clip if gradients are exploding

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def train_epoch(train_dataloader, generator, discriminator, optimizer_g, optimizer_d, content_loss_criterion, adversarial_loss_criterion, truncated_vgg19, grad_clip = None):
    """
    Epoch trainer

    train_dataloader: train dataloader
    generator: generator
    discriminator: discriminator
    optimizer_g: optimizer for the generator
    optimizer_d: optimizer for the discriminator
    content_loss_criterion: content loss function
    adversarial_loss_criterion: adversarial loss function
    truncated_vgg19: truncated VGG19 network
    grad_clip: gradient clipping
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalisation

    # MSE Loss
    if content_loss_criterion == "MSE":
        content_loss_criterion = nn.MSELoss()
    # MAE Loss
    if content_loss_criterion == "MAE":
        content_loss_criterion = nn.L1Loss()
    # Huber Loss
    if content_loss_criterion == "huber":
        content_loss_criterion = nn.HuberLoss()

    # BCE Loss
    if adversarial_loss_criterion == "BCE":
        adversarial_loss_function = nn.BCEWithLogitsLoss()
    # MSE Loss
    if adversarial_loss_criterion == "MSE":
        adversarial_loss_function = nn.MSELoss()

    # Keep track of Losses/PSNR/SSIM
    total_content_loss = 0
    total_adversarial_g_loss = 0
    total_perceptual_loss = 0
    total_adversarial_d_loss = 0
    total_psnr = 0
    total_ssim = 0

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_dataloader):
        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        ### GENERATOR UPDATE ###

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        sr_imgs = convert_image(sr_imgs, source = "[-1, 1]", target = "imagenet-norm")  # (N, 3, 96, 96), imagenet-normed

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they"re constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the perceptual loss using the content and adversarial losses
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

        # Binary Cross-Entropy loss
        if adversarial_loss_criterion == "BCE":
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.ones_like(sr_discriminated))
        # MSE loss
        if adversarial_loss_criterion == "MSE":
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.ones_like(sr_discriminated))
        # Wasserstein loss
        elif adversarial_loss_criterion == "wasserstein":
            adversarial_loss = -1 * sr_discriminated.mean()
        # Hinge loss
        elif adversarial_loss_criterion == "hinge":
            adversarial_loss = -1 * sr_discriminated.mean()

        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        sr_imgs_y = convert_image(sr_imgs, source = "imagenet-norm", target = "[0, 255]")
        hr_imgs_y = convert_image(hr_imgs, source = "imagenet-norm", target = "[0, 255]")

        # Keep track of loss
        total_content_loss += float(content_loss.item())
        total_adversarial_g_loss += float(adversarial_loss.item())
        total_perceptual_loss += float(perceptual_loss.item())

        # Keep track of PSNR
        total_psnr += float(piq.psnr(sr_imgs_y, hr_imgs_y, data_range = 255.0))

        # Keep track of SSIM
        total_ssim += float(piq.ssim(sr_imgs_y, hr_imgs_y, data_range = 255.0, downsample = True))

        # Log last image to wandb
        if i == len(train_dataloader) - 1:
            hr_img_grid = make_grid(hr_imgs_y, normalize = True)
            wandb.log({"High-Resolution Images": [wandb.Image(hr_img_grid)]})

            sr_img_grid = make_grid(sr_imgs_y, normalize = True)
            wandb.log({"Super-Resolution Images": [wandb.Image(sr_img_grid)]})
        
            del hr_img_grid, sr_img_grid

        ### DISCRIMINATOR UPDATE ###

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())

        # Binary Cross-Entropy loss
        if adversarial_loss_criterion == "BCE":
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.zeros_like(sr_discriminated)) + adversarial_loss_function(hr_discriminated, torch.ones_like(hr_discriminated))
        # MSE loss
        if adversarial_loss_criterion == "MSE":
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.zeros_like(sr_discriminated)) + adversarial_loss_function(hr_discriminated, torch.ones_like(hr_discriminated))
        # Wasserstein loss
        elif adversarial_loss_criterion == "wasserstein":
            adversarial_loss = sr_discriminated.mean() - hr_discriminated.mean()
        # Hinge loss
        elif adversarial_loss_criterion == "hinge":
            adversarial_loss = nn.ReLU()(1 - hr_discriminated).mean() + nn.ReLU()(1 + sr_discriminated).mean()

        # Back propagation
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        total_adversarial_d_loss += float(adversarial_loss.item())

        del lr_imgs, hr_imgs, sr_imgs, sr_imgs_y, hr_imgs_y, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    return total_content_loss / len(train_dataloader), total_adversarial_g_loss / len(train_dataloader), total_perceptual_loss / len(train_dataloader), total_adversarial_d_loss / len(train_dataloader), total_psnr / len(train_dataloader), total_ssim / len(train_dataloader)

def train(train_dataloader, generator, discriminator, optimizer_g, optimizer_d, content_loss_criterion, adversarial_loss_criterion, iterations, logger, VGG_params = (5, 4), starting_epoch = 1, grad_clip = None, checkpoint = None):
    """
    Training
    """

    # Adam
    if (optimizer_g == "adam") and (checkpoint == None):
        optimizer_g = torch.optim.Adam(params = filter(lambda p: p.requires_grad, generator.parameters()), lr = learning_rate)
    # SGD Nesterov
    if (optimizer_g == "sgd-n") and (checkpoint == None):
        optimizer_g = torch.optim.SGD(params = filter(lambda p: p.requires_grad, generator.parameters()), lr = learning_rate, nesterov = True, momentum = 0.9)
    # SGD
    if (optimizer_g == "sgd") and (checkpoint == None):
        optimizer_g = torch.optim.SGD(params = filter(lambda p: p.requires_grad, generator.parameters()), lr = learning_rate)

    # Adam
    if (optimizer_d == "adam") and (checkpoint == None):
        optimizer_d = torch.optim.Adam(params = filter(lambda p: p.requires_grad, discriminator.parameters()), lr = learning_rate)
    # SGD Nesterov
    if (optimizer_d == "sgd-n") and (checkpoint == None):
        optimizer_d = torch.optim.SGD(params = filter(lambda p: p.requires_grad, discriminator.parameters()), lr = learning_rate, nesterov = True, momentum = 0.9)
    # SGD
    if (optimizer_d == "sgd") and (checkpoint == None):
        optimizer_d = torch.optim.SGD(params = filter(lambda p: p.requires_grad, discriminator.parameters()), lr = learning_rate)

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Apply VGG (if desired)
    truncated_vgg19 = TruncatedVGG19(i = VGG_params[0], j = VGG_params[1])
    truncated_vgg19.eval()

    # Total number of epochs to train for (based on iterations)
    epochs = int(iterations // len(train_dataloader))

    # Epochs
    for epoch in range(starting_epoch, epochs + 1):
        # At the halfway point, reduce learning rate by a tenth
        if epoch == int(epochs // 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # Train and validation epoch
        total_content_loss, total_adversarial_g_loss, total_perceptual_loss, total_adversarial_d_loss, train_psnr, train_ssim = train_epoch(train_dataloader, generator = generator, discriminator = discriminator, optimizer_g = optimizer_g, optimizer_d = optimizer_d, content_loss_criterion = content_loss_criterion, adversarial_loss_criterion = adversarial_loss_criterion, truncated_vgg19 = truncated_vgg19, grad_clip = grad_clip)

        print(f"Epoch: {epoch} / {epochs}, Content Loss {total_content_loss}, Adversarial Generator Loss {total_adversarial_g_loss}, Perceptual Loss {total_perceptual_loss}, Adversarial Discriminator Loss {total_adversarial_d_loss}, PSNR {train_psnr}, SSIM {train_ssim}")

        logger.log({"total_content_loss": total_content_loss})
        logger.log({"total_adversarial_g_loss": total_adversarial_g_loss})
        logger.log({"total_perceptual_loss": total_perceptual_loss})
        logger.log({"total_adversarial_d_loss": total_adversarial_d_loss})
        logger.log({"train_psnr": train_psnr})
        logger.log({"train_ssim": train_ssim})

        # Create checkpoint folder
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        # Save the model checkpoint
        checkpoint_name = f"checkpoint_srgan.pth.tar"
        checkpoint_path = os.path.join("checkpoints", checkpoint_name)

        # Save checkpoint
        torch.save({"epoch": epoch,
                    "generator": generator,
                    "discriminator": discriminator,
                    "optimizer_g": optimizer_g,
                    "optimizer_d": optimizer_d,
                    "total_content_loss": total_content_loss,
                    "total_adversarial_g_loss": total_adversarial_g_loss,
                    "total_perceptual_loss": total_perceptual_loss,
                    "total_adversarial_d_loss": total_adversarial_d_loss,
                    "train_psnr": train_psnr,
                    "train_ssim": train_ssim},
                    checkpoint_path)
        
    return

def main():
    # Set random seed for reproducibility
    randomer = 50
    torch.manual_seed(randomer)
    torch.cuda.manual_seed_all(randomer)
    random.seed(randomer)
    np.random.seed(randomer)

    # Read settings from the YAML file
    #args = parse_arguments()
    #settings = read_settings(args.config)

    # Access and use the settings as needed
    #model_settings = settings.get("model", {})
    #train_settings = settings.get("train", {})
    #print(model_settings)
    #print(train_settings)

    # Initialise 'wandb' for logging
    wandb_logger = Logger(f"inm705_SRGAN_Adam-x2_MSE", project = "inm705_cwk")
    logger = wandb_logger.get_logger()

    # Custom dataloaders
    train_dataset = SRDataset(data_folder, split = "train", crop_size = crop_size, scaling_factor = scaling_factor, lr_img_type = "imagenet-norm", hr_img_type = "imagenet-norm")
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True)  # note that we're passing the collate function here

    checkpoint = None # train from scratch, without a checkpoint
    #checkpoint = "checkpoints/checkpoint_srgan.pth.tar" # use if wanting to train from a checkpoint

    if checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size = large_kernel_size_g, small_kernel_size = small_kernel_size_g, n_channels = n_channels_g, n_blocks = n_blocks_g, scaling_factor = scaling_factor)

        # Initialise generator network with pretrained SRResNet
        generator.initialise_with_srresnet(srresnet_checkpoint = srresnet_checkpoint)

        # Discriminator
        discriminator = Discriminator(kernel_size = kernel_size_d, n_channels = n_channels_d, n_blocks = n_blocks_d, fc_size = fc_size_d)

        train(train_dataloader, generator = generator, discriminator = discriminator, optimizer_g = "adam", optimizer_d = "adam", content_loss_criterion = "MSE", adversarial_loss_criterion = "MSE", iterations = iterations, logger = logger, VGG_params = (5, 4), starting_epoch = 1, grad_clip = grad_clip, checkpoint = None)

    else:
        checkpoint = torch.load(checkpoint)
        starting_epoch = checkpoint["epoch"] + 1
        generator = checkpoint["generator"]
        discriminator = checkpoint["discriminator"]
        optimizer_g = checkpoint["optimizer_g"]
        optimizer_d = checkpoint["optimizer_d"]
        train(train_dataloader, generator = generator, discriminator = discriminator, optimizer_g = optimizer_g, optimizer_d = optimizer_d, content_loss_criterion = "MSE", adversarial_loss_criterion = "MSE", iterations = iterations, logger = logger, VGG_params = (5, 4), starting_epoch = starting_epoch, grad_clip = grad_clip, checkpoint = checkpoint)

if __name__ == "__main__":
    main()