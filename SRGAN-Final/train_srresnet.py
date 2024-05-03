import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import piq

from models import SRResNet
from datasets import SRDataset
from utils import *
from logger import Logger
from loss import TruncatedVGG19

## As found in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution ##
## With some changes (residual scaling, adding GELU, attention, disabling batch normalisation, etc.) ##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def train_epoch(train_dataloader, model, criterion, optimizer, truncated_vgg19 = None, with_VGG = False, grad_clip = None):
    """
    Epoch trainer

    train_dataloader: DataLoader for training data
    model: model
    criterion: content loss function
    optimizer: optimizer
    truncated_vgg19: truncated VGG19 model for feature extraction
    with_VGG: flag indicating whether to use VGG feature loss
    grad_clip: gradient clipping
    """
    model.train()  # training mode enables batch normalisation
    
    # MSE Loss
    if criterion == "MSE":
        loss_function = nn.MSELoss()
    # MAE Loss
    if criterion == "MAE":
        loss_function = nn.L1Loss()
    # SSIM Loss
    if criterion == "SSIM":
        loss_function = piq.SSIMLoss(downsample = True, data_range = 255.0)

    if with_VGG == True:
        loss_function = nn.MSELoss()
        print("Defaulting to MSE loss for VGG...")

    # Keep track of Loss/PSNR/SSIM
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_dataloader):

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed, 24x24 e.g.
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1], 96x96 e.g.

        # Forward propagation
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Calculate VGG feature maps for SR and HR images
        if with_VGG == True:
            sr_imgs_norm = convert_image(sr_imgs, source = "[-1, 1]", target = "imagenet-norm")  # (N, 3, 96, 96), imagenet-normed

            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs_norm)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            # Loss
            loss = loss_function(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            del sr_imgs_norm, sr_imgs_in_vgg_space, hr_imgs_in_vgg_space

        else:
            if criterion == "SSIM":
                sr_imgs_y = convert_image(sr_imgs, source = "[-1, 1]", target = "[0, 255]")
                hr_imgs_y = convert_image(hr_imgs, source = "[-1, 1]", target = "[0, 255]")

                # Loss
                loss = loss_function(sr_imgs_y, hr_imgs_y)

                del sr_imgs_y, hr_imgs_y
            else:
                # Loss
                loss = loss_function(sr_imgs, hr_imgs)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if (isinstance(grad_clip, int) == True) or (isinstance(grad_clip, float) == True):
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        sr_imgs_y = convert_image(sr_imgs, source = "[-1, 1]", target = "[0, 255]")
        hr_imgs_y = convert_image(hr_imgs, source = "[-1, 1]", target = "[0, 255]")

        # Keep track of loss
        total_loss += float(loss.item())

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
        
        del lr_imgs, hr_imgs, sr_imgs, sr_imgs_y, hr_imgs_y

    del hr_img_grid, sr_img_grid

    return total_loss / len(train_dataloader), total_psnr / len(train_dataloader), total_ssim / len(train_dataloader)

def validate_epoch(val_dataloader, model, criterion, truncated_vgg19 = None, with_VGG = False):
    """
    Epoch validator

    val_dataloader: DataLoader for validation data
    model: model
    criterion: content loss function
    truncated_vgg19: truncated VGG19 model for feature extraction
    with_VGG: flag indicating whether to use VGG feature loss
    """
    model.eval()  # Evaluation mode, no batch norm

    # MSE Loss
    if criterion == "MSE":
        loss_function = nn.MSELoss()
    # MAE Loss
    if criterion == "MAE":
        loss_function = nn.L1Loss()
    # SSIM Loss
    if criterion == "SSIM":
        loss_function = piq.SSIMLoss(downsample = True, data_range = 255.0)

    if with_VGG == True:
        loss_function = nn.MSELoss()
        print("Defaulting to MSE loss for VGG...")

    # Keep track of Loss/PSNR/SSIM
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():  # No gradient computation during validation
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(val_dataloader):

            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed, 24x24 e.g.
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1], 96x96 e.g.

            # Forward propagation
            sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

            # Calculate VGG feature maps for SR and HR images
            if with_VGG:
                sr_imgs_norm = convert_image(sr_imgs, source = "[-1, 1]", target = "imagenet-norm")  # (N, 3, 96, 96), imagenet-normed

                sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs_norm)
                hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

                # Loss
                loss = loss_function(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

                del sr_imgs_norm, sr_imgs_in_vgg_space, hr_imgs_in_vgg_space

            else:
                if criterion == "SSIM":
                    sr_imgs_y = convert_image(sr_imgs, source = "[-1, 1]", target = "[0, 255]")
                    hr_imgs_y = convert_image(hr_imgs, source = "[-1, 1]", target = "[0, 255]")

                    # Loss
                    loss = loss_function(sr_imgs_y, hr_imgs_y)

                    del sr_imgs_y, hr_imgs_y
                else:
                    # Loss
                    loss = loss_function(sr_imgs, hr_imgs)
            
            sr_imgs_y = convert_image(sr_imgs, source = "[-1, 1]", target = "[0, 255]")
            hr_imgs_y = convert_image(hr_imgs, source = "[-1, 1]", target = "[0, 255]")

            # Keep track of loss
            total_loss += float(loss.item())

            # Keep track of PSNR
            total_psnr += float(piq.psnr(sr_imgs_y, hr_imgs_y, data_range = 255.0))

            # Keep track of SSIM
            total_ssim += float(piq.ssim(sr_imgs_y, hr_imgs_y, data_range = 255.0, downsample = True))

            del lr_imgs, hr_imgs, sr_imgs, sr_imgs_y, hr_imgs_y

    return total_loss / len(val_dataloader), total_psnr / len(val_dataloader), total_ssim / len(val_dataloader)

def train(train_dataloader, val_dataloader, model, iterations, logger, with_VGG = False, VGG_params = (5, 4), criterion = "MSE", starting_epoch = 1, optimizer = "adam", learning_rate = 0.0001, adaptive_lr = False, grad_clip = None, dcgan_weight_init = False, checkpoint = None):
    """
    Training
    """
    val_losses = [float("inf")]
    counter = 0

    # Adam
    if (optimizer == "adam") and (checkpoint == None):
        optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    # SGD Nesterov
    if (optimizer == "sgd-n") and (checkpoint == None):
        optimizer = torch.optim.SGD(params = filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, nesterov = True, momentum = 0.9)
    # SGD
    if (optimizer == "sgd") and (checkpoint == None):
        optimizer = torch.optim.SGD(params = filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)

    # Move to default device
    model = model.to(device)

    # Apply VGG (if desired)
    truncated_vgg19 = TruncatedVGG19(i = int(VGG_params[0]), j = int(VGG_params[1]))
    truncated_vgg19.eval()

    # Weight initialise with N(0, 0.02)
    if dcgan_weight_init == True:
        model.apply(weights_init)

    # Total number of epochs to train for (based on iterations)
    epochs = int(iterations // len(train_dataloader))

    # Epochs
    for epoch in range(starting_epoch, epochs + 1):
        # At the halfway point, reduce learning rate by a tenth
        if (epoch == int(epochs // 2)) and (adaptive_lr == True):
            adjust_learning_rate(optimizer, 0.1)

        # Train and validation epoch
        train_loss, train_psnr, train_ssim = train_epoch(train_dataloader, model = model, criterion = criterion, optimizer = optimizer, truncated_vgg19 = truncated_vgg19, with_VGG = with_VGG, grad_clip = grad_clip)
        val_loss, val_psnr, val_ssim = validate_epoch(val_dataloader, model = model, criterion = criterion, truncated_vgg19 = truncated_vgg19, with_VGG = with_VGG)

        print(f"Epoch: {epoch} / {epochs}, Train Loss {train_loss}, Validation Loss {val_loss}, Train PSNR {train_psnr}, Validation PSNR {val_psnr}, Train SSIM {train_ssim}, Validation SSIM {val_ssim}")

        val_losses.append(val_loss)

        logger.log({"train_loss": train_loss})
        logger.log({"validation_loss": val_loss})
        logger.log({"train_psnr": train_psnr})
        logger.log({"validation_psnr": val_psnr})
        logger.log({"train_ssim": train_ssim})
        logger.log({"validation_ssim": val_ssim})

        if (val_loss + 0.00000001) < val_losses[-2]:
            # Restart patience (improvement in validation loss)
            counter = 0

            # Create checkpoint folder
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            # Save the model checkpoint
            checkpoint_name = f"checkpoint_srresnet.pth.tar"
            checkpoint_path = os.path.join("checkpoints", checkpoint_name)

            # Save checkpoint
            torch.save({"epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_psnr": train_psnr,
                        "val_psnr": val_psnr,
                        "train_ssim": train_ssim,
                        "val_ssim": val_ssim},
                        checkpoint_path)

        elif (val_loss + 0.00000001) > val_losses[-2]:
            # Add one to patience
            counter += 1

            if val_loss < val_losses[-2]:
                # Create checkpoint folder
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")

                checkpoint_name = f"checkpoint_srresnet.pth.tar"
                checkpoint_path = os.path.join("checkpoints", checkpoint_name)

                # Save checkpoint
                torch.save({"epoch": epoch,
                            "model": model,
                            "optimizer": optimizer,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_psnr": train_psnr,
                            "val_psnr": val_psnr,
                            "train_ssim": train_ssim,
                            "val_ssim": val_ssim},
                            checkpoint_path)

            # Patience reached, stop training (no significant improvement in validation loss after 5 epochs)
            if counter >= 5:
                print("Ending training due to lack of improvement...")
                break

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
    srresnet_settings = settings.get("SRResNet", {})

    # Initialise 'wandb' for logging
    wandb_logger = Logger(logger_settings["logger_name"], project = logger_settings["project_name"])
    logger = wandb_logger.get_logger()

    # Custom dataloaders
    train_dataset = SRDataset(data_folder = data_settings["data_folder"], split = "train", crop_size = data_settings["crop_size"], scaling_factor = data_settings["scaling_factor"], lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]")
    train_dataloader = DataLoader(train_dataset, batch_size = srresnet_settings["batch_size"], shuffle = True, num_workers = srresnet_settings["workers"], pin_memory = True)  # note that we are passing the collate function here

    val_dataset = SRDataset(data_folder = data_settings["data_folder"], split = "val", crop_size = 0, scaling_factor = data_settings["scaling_factor"], lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]")
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True, num_workers = srresnet_settings["workers"], pin_memory = True)  # note that we are passing the collate function here

    checkpoint = None
    starting_epoch = srresnet_settings["starting_epoch"]
    optimizer = srresnet_settings["optimizer"]

    if srresnet_settings["checkpoint"].lower() == "none":
        model = SRResNet(large_kernel_size = srresnet_settings["large_kernel_size"], small_kernel_size = srresnet_settings["small_kernel_size"], n_channels = srresnet_settings["n_channels"], n_blocks = srresnet_settings["n_blocks"], scaling_factor = data_settings["scaling_factor"], activation = srresnet_settings["activation"], enable_standard_bn = srresnet_settings["enable_standard_bn"], resid_scale_factor = srresnet_settings["resid_scale_factor"], self_attention = srresnet_settings["self_attention"])
    else:
        checkpoint = torch.load(srresnet_settings["checkpoint"], map_location = device)
        starting_epoch = checkpoint["epoch"] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    train(train_dataloader, val_dataloader, model, srresnet_settings["iterations"], logger, with_VGG = srresnet_settings["with_VGG"], VGG_params = srresnet_settings["VGG_params"], criterion = srresnet_settings["criterion"], starting_epoch = starting_epoch, optimizer = optimizer, learning_rate = srresnet_settings["learning_rate"], adaptive_lr = srresnet_settings["adaptive_lr"], grad_clip = srresnet_settings["grad_clip"], dcgan_weight_init = srresnet_settings["dcgan_weight_init"], checkpoint = checkpoint)

if __name__ == "__main__":
    main()
