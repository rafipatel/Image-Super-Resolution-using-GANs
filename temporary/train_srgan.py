import time
from torch import nn
from models import Generator, Discriminator
from loss import TruncatedVGG19
from datasets import SRDataset
from utils import *

# Data parameters
data_folder = "./"  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch):
    """
    One epoch"s training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        sr_imgs = convert_image(sr_imgs, source="[-1, 1]", target="imagenet-norm")  # (N, 3, 96, 96), imagenet-normed

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they"re constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn"t we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we"d be back-propagating (finding gradients) over the G too when backward() is called
        # It"s actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print("Epoch: [{0}][{1}/{2}]----"
                  "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----"
                  "Data Time {data_time.val:.3f} ({data_time.avg:.3f})----"
                  "Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----"
                  "Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----"
                  "Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})".format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored

def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)

        # Initialize generator network with pretrained SRResNet
        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

        # Initialize generator"s optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr)

        # Discriminator
        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)

        # Initialize discriminator"s optimizer
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        generator = checkpoint["generator"]
        discriminator = checkpoint["discriminator"]
        optimizer_g = checkpoint["optimizer_g"]
        optimizer_d = checkpoint["optimizer_d"]
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint["epoch"] + 1))

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split="train",
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type="imagenet-norm",
                              hr_img_type="imagenet-norm")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch"s training
        train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              truncated_vgg19=truncated_vgg19,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch)

        # Save checkpoint
        torch.save({"epoch": epoch,
                    "generator": generator,
                    "discriminator": discriminator,
                    "optimizer_g": optimizer_g,
                    "optimizer_d": optimizer_d},
                    "checkpoint_srgan.pth.tar")

def train(train_dataloader, val_dataloader, model, iterations, logger, with_VGG = False, VGG_params = (5, 4), criterion = "MSE", starting_epoch = 1, optimizer = "adam", checkpoint = None):
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
    truncated_vgg19 = TruncatedVGG19(i = VGG_params[0], j = VGG_params[1])
    truncated_vgg19.eval()

    # Total number of epochs to train for (based on iterations)
    epochs = int(iterations // len(train_dataloader))

    # Epochs
    for epoch in range(starting_epoch, epochs + 1):
        # At the halfway point, reduce learning rate by a tenth
        if epoch == int(epochs // 2):
            adjust_learning_rate(optimizer, 0.1)

        # Train and validation epoch
        train_loss, train_psnr, train_ssim = train_epoch(train_dataloader, model = model, criterion = criterion, optimizer = optimizer, truncated_vgg19 = truncated_vgg19, with_VGG = with_VGG)
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
    #args = parse_arguments()
    #settings = read_settings(args.config)

    # Access and use the settings as needed
    #model_settings = settings.get("model", {})
    #train_settings = settings.get("train", {})
    #print(model_settings)
    #print(train_settings)

    # Initialise 'wandb' for logging
    wandb_logger = Logger(f"inm705_SRResNet_Adam_MAE_adaptive_lr_no_batch_norm_gelu", project = "inm705_cwk")
    logger = wandb_logger.get_logger()

    # Custom dataloaders
    train_dataset = SRDataset(data_folder, split = "train", crop_size = crop_size, scaling_factor = scaling_factor, lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]")
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True)  # note that we're passing the collate function here

    val_dataset = SRDataset(data_folder, split = "val", crop_size = 0, scaling_factor = scaling_factor, lr_img_type = "imagenet-norm", hr_img_type = "[-1, 1]")
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True, num_workers = workers, pin_memory = True)  # note that we're passing the collate function here

    checkpoint = None

    if checkpoint is None:
        model = 
        train(train_dataloader, val_dataloader, model, iterations, logger, with_VGG = False, VGG_params = (5, 4), criterion = "MAE", starting_epoch = 1, optimizer = "adam", checkpoint = None)

    else:
        checkpoint = torch.load(checkpoint)
        starting_epoch = checkpoint["epoch"] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        train(train_dataloader, val_dataloader, model, iterations, logger, VGG_params = (5, 4), criterion = "MAE", starting_epoch = starting_epoch, optimizer = optimizer, checkpoint = checkpoint)

if __name__ == "__main__":
    main()