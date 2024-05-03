import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import piq
from torcheval.metrics import FrechetInceptionDistance

from models import Generator, Discriminator
from datasets import SRDataset
from utils import *
from logger import Logger
from loss import TruncatedVGG19, BCEWithLogitsLoss_label_smoothing

## As found in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution ##
## With some changes (residual scaling, adding GELU, attention, disabling batch normalisation, etc.) ##

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def train_epoch(train_dataloader, generator, discriminator, optimizer_g, optimizer_d, content_loss_criterion, adversarial_loss_criterion, truncated_vgg19, beta = 0.001, grad_clip = None):
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
    # BCE with label smoothing Loss
    if adversarial_loss_criterion == "BCE_label_smoothing":
        adversarial_loss_function = BCEWithLogitsLoss_label_smoothing(label_smoothing = 0.1)
    # MSE Loss
    if adversarial_loss_criterion == "MSE":
        adversarial_loss_function = nn.MSELoss()

    # Keep track of Losses/PSNR/SSIM/FID
    total_content_loss = 0
    total_adversarial_g_loss = 0
    total_perceptual_loss = 0
    total_adversarial_d_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    fid = FrechetInceptionDistance().to(device)

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

        # Wasserstein loss
        if adversarial_loss_criterion == "wasserstein":
            adversarial_loss = -1 * sr_discriminated.mean()
        # Hinge loss
        elif adversarial_loss_criterion == "hinge":
            adversarial_loss = -1 * sr_discriminated.mean()
        # Other losses
        else:
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.ones_like(sr_discriminated))

        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if (isinstance(grad_clip, int) == True) or (isinstance(grad_clip, float) == True):
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

        # Wasserstein loss
        if adversarial_loss_criterion == "wasserstein":
            adversarial_loss = sr_discriminated.mean() - hr_discriminated.mean()
        # Hinge loss
        elif adversarial_loss_criterion == "hinge":
            adversarial_loss = nn.ReLU()(1 - hr_discriminated).mean() + nn.ReLU()(1 + sr_discriminated).mean()
        # Other losses
        else:
            adversarial_loss = adversarial_loss_function(sr_discriminated, torch.zeros_like(sr_discriminated)) + adversarial_loss_function(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back propagation
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if (isinstance(grad_clip, int) == True) or (isinstance(grad_clip, float) == True):
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        total_adversarial_d_loss += float(adversarial_loss.item())

        # FID update
        fid.update(convert_image(hr_imgs, source = "imagenet-norm", target = "[0, 1]"), is_real = True)
        fid.update(convert_image(sr_imgs, source = "imagenet-norm", target = "[0, 1]"), is_real = False)

        del lr_imgs, hr_imgs, sr_imgs, sr_imgs_y, hr_imgs_y, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    return total_content_loss / len(train_dataloader), total_adversarial_g_loss / len(train_dataloader), total_perceptual_loss / len(train_dataloader), total_adversarial_d_loss / len(train_dataloader), total_psnr / len(train_dataloader), total_ssim / len(train_dataloader), float(fid.compute())

def train(train_dataloader, generator, discriminator, optimizer_g, optimizer_d, content_loss_criterion, adversarial_loss_criterion, iterations, logger, learning_rate = 0.0001, VGG_params = (5, 4), starting_epoch = 1, grad_clip = None, beta = 0.001, dcgan_weight_init_d = False, checkpoint = None):
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

    # Weight initialise with N(0, 0.02) - discriminator only
    if dcgan_weight_init_d == True:
        discriminator.apply(weights_init)

    # Total number of epochs to train for (based on iterations)
    epochs = int(iterations // len(train_dataloader))

    # Epochs
    for epoch in range(starting_epoch, epochs + 1):
        # At the halfway point, reduce learning rate by a tenth
        if epoch == int(epochs // 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # Train and validation epoch
        total_content_loss, total_adversarial_g_loss, total_perceptual_loss, total_adversarial_d_loss, train_psnr, train_ssim, fid = train_epoch(train_dataloader, generator = generator, discriminator = discriminator, optimizer_g = optimizer_g, optimizer_d = optimizer_d, content_loss_criterion = content_loss_criterion, adversarial_loss_criterion = adversarial_loss_criterion, truncated_vgg19 = truncated_vgg19, beta = beta, grad_clip = grad_clip)

        print(f"Epoch: {epoch} / {epochs}, Content Loss {total_content_loss}, Adversarial Generator Loss {total_adversarial_g_loss}, Perceptual Loss {total_perceptual_loss}, Adversarial Discriminator Loss {total_adversarial_d_loss}, PSNR {train_psnr}, SSIM {train_ssim}, FID {fid}")

        logger.log({"total_content_loss": total_content_loss})
        logger.log({"total_adversarial_g_loss": total_adversarial_g_loss})
        logger.log({"total_perceptual_loss": total_perceptual_loss})
        logger.log({"total_adversarial_d_loss": total_adversarial_d_loss})
        logger.log({"train_psnr": train_psnr})
        logger.log({"train_ssim": train_ssim})
        logger.log({"FID": fid})

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
                    "train_ssim": train_ssim,
                    "FID": fid},
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
    args = parse_arguments()
    settings = read_settings(args.config)

    # Access and use the settings as needed
    data_settings = settings.get("Data", {})
    logger_settings = settings.get("Logger", {})
    srgan_settings = settings.get("SRGAN", {})

    # Initialise 'wandb' for logging
    wandb_logger = Logger(logger_settings["logger_name"], project = logger_settings["project_name"])
    logger = wandb_logger.get_logger()

    # Custom dataloaders
    train_dataset = SRDataset(data_folder = data_settings["data_folder"], split = "train", crop_size = data_settings["crop_size"], scaling_factor = data_settings["scaling_factor"], lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]")
    train_dataloader = DataLoader(train_dataset, batch_size = srgan_settings["batch_size"], shuffle = True, num_workers = srgan_settings["workers"], pin_memory = True)  # note that we are passing the collate function here

    checkpoint = None
    starting_epoch = srgan_settings["starting_epoch"]
    optimizer_g = srgan_settings["optimizer_g"]
    optimizer_d = srgan_settings["optimizer_d"]

    if srgan_settings["checkpoint"].lower() == "none":
        # Generator
        generator = Generator(large_kernel_size = srgan_settings["large_kernel_size_g"], small_kernel_size = srgan_settings["small_kernel_size_g"], n_channels = srgan_settings["n_channels_g"], n_blocks = srgan_settings["n_blocks_g"], scaling_factor = data_settings["scaling_factor"], activation = srgan_settings["activation_g"], enable_standard_bn = srgan_settings["enable_standard_bn_g"], resid_scale_factor = srgan_settings["resid_scale_factor"], self_attention = srgan_settings["self_attention_g"])
        
        # Initialise generator network with pretrained SRResNet
        if srgan_settings["srresnet_checkpoint"].lower() == "none":
            generator.initialise_with_srresnet(srresnet_checkpoint = srgan_settings["srresnet_checkpoint"])

        # Discriminator
        discriminator = Discriminator(kernel_size = srgan_settings["kernel_size_d"], n_channels = srgan_settings["n_channels_d"], n_blocks = srgan_settings["n_blocks_d"], fc_size = srgan_settings["fc_size_d"], self_attention = srgan_settings["self_attention_d"], spectral_norm = srgan_settings["spectral_norm_d"])

    else:
        checkpoint = torch.load(checkpoint, map_location = device)
        starting_epoch = checkpoint["epoch"] + 1
        generator = checkpoint["generator"]
        discriminator = checkpoint["discriminator"]
        optimizer_g = checkpoint["optimizer_g"]
        optimizer_d = checkpoint["optimizer_d"]

    train(train_dataloader, generator = generator, discriminator = discriminator, optimizer_g = optimizer_g, optimizer_d = optimizer_d, content_loss_criterion = srgan_settings["content_loss_criterion"], adversarial_loss_criterion = srgan_settings["adversarial_loss_criterion"], iterations = srgan_settings["iterations"], logger = logger, VGG_params = srgan_settings["VGG_params"], starting_epoch = starting_epoch, grad_clip = srgan_settings["grad_clip"], beta = srgan_settings["beta"], dcgan_weight_init_d = srgan_settings["dcgan_weight_init_d"], checkpoint = None)

if __name__ == "__main__":
    main()
