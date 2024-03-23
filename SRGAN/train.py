import torch
import config
from torch import nn
from torch import optim
# from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import SuperResolutionDataset
from logger import Logger
import numpy as np
import random
import wandb

#torch.backends.cudnn.benchmark = True

SRResnet = True

def train_epoch(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    
    # loop = tqdm(loader, leave = True)

    
    for idx, (low_res, high_res) in enumerate(loader):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        if SRResnet:
            fake = gen(low_res)
            l2_loss = mse(fake, high_res)
            gen_loss =  l2_loss # loss_for_vgg + adversarial_loss
        
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            return gen_loss.item()
        
        else:
            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            ) #one sided label smoothing (extra), rest is as per paper

            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            print('=' * 50)
            #print(disc_loss_fake)
            loss_disc = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            disc_fake = disc(fake)
            l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake)) #to play around with loss terms
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss =  loss_for_vgg + adversarial_loss
            
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            # wandb.log({"gen_loss":gen_loss.item(),"VGG loss":loss_for_vgg,"Disc_loss":loss_disc.item(), "MSE":l2_loss.item(),"Adverarial_loss": adversarial_loss.item()})

            return gen_loss.item(), loss_for_vgg, loss_disc.item(), l2_loss.item(), adversarial_loss.item()

def train(train_dataloader, logger, in_channels = 3, optimizer = "adam"):
    gen = Generator(in_channels = in_channels).to(config.DEVICE)
    disc = Discriminator(in_channels = in_channels).to(config.DEVICE)

    if optimizer == "adam":
        opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas = (0.9, 0.999))
        opt_disc = optim.Adam(disc.parameters(), lr = config.LEARNING_RATE, betas = (0.9, 0.999))
    elif optimizer == "radam":
        opt_gen = optim.RAdam(gen.parameters(), lr = config.LEARNING_RATE, betas = (0.9, 0.999))
        opt_disc = optim.RAdam(disc.parameters(), lr = config.LEARNING_RATE, betas = (0.9, 0.999))

    mse_loss_function = nn.MSELoss()
    bce_loss_function = nn.BCEWithLogitsLoss()
    vgg_loss_function = VGGLoss()

    if config.LOAD_MODEL:
        print("=> Loading checkpoint")
        checkpoint = torch.load("../gen_100_epochs.tar")
        gen.load_state_dict(checkpoint["state_dict"])
        opt_gen.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in opt_gen.param_groups:
        param_group["lr"] = config.LEARNING_RATE

    for epoch in range(config.NUM_EPOCHS):

        if SRResnet:
            gen_loss = train_epoch(train_dataloader, disc, gen, opt_gen, opt_disc, mse_loss_function, bce_loss_function, vgg_loss_function)

            logger.log({'gen_loss(only)': gen_loss})

            print(f"Epoch: {epoch} / {config.NUM_EPOCHS}, Generator Loss {gen_loss}")

        else:
            gen_loss, vgg_loss, disc_loss, l2_loss, adver_loss = train_epoch(train_dataloader, disc, gen, opt_gen, opt_disc, mse_loss_function, bce_loss_function, vgg_loss_function)

            logger.log({'gen_loss(includingVGGloss)': gen_loss,
                    'vgg_loss' : vgg_loss,
                    'disc_loss': disc_loss,
                    'mse_loss': l2_loss, 
                    'adversarial_loss': adver_loss})
            
            print(f"Epoch: {epoch} / {config.NUM_EPOCHS}, Generator Loss {gen_loss}, VGG Loss {vgg_loss}, Discriminator Loss {disc_loss}, L2 Loss {l2_loss}, Adversarial Loss {adver_loss}")

        if epoch > 0 and (epoch % 500) == 0:
          if config.SAVE_MODEL:
              
            print("=> Saving checkpoint Generator")
            checkpoint = {
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            
            if SRResnet:
                torch.save(checkpoint, f'genSRResnet_{epoch}_epochs.tar')

            else:
                torch.save(checkpoint, f'gen_{epoch}_epochs.tar')

                print("=> Saving checkpoint Discriminator")
                checkpoints = {
                    "state_dict": disc.state_dict(),
                    "optimizer": opt_disc.state_dict(),
                }
                torch.save(checkpoints, f'disc_{epoch}_epochs.tar')

    


def main():
    # Set random seed for reproducibility
    randomer = 50
    torch.manual_seed(randomer)
    torch.cuda.manual_seed_all(randomer)
    random.seed(randomer)
    np.random.seed(randomer)

    # Initialise "wandb" for logging
    wandb_logger = Logger(f"inm705_SRResnet", project = "inm705_RESNET")
    logger = wandb_logger.get_logger()

    # dataset = SuperResolutionDataset(root_dir = "../DIV2K_train_HR/")
    # dataset = SuperResolutionDataset(root_dir = "E:\\GAN\\")
    dataset = SuperResolutionDataset(root_dir = "/users/adfx757/GAN/")
    train_dataloader = DataLoader(dataset, batch_size = config.BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = config.NUM_WORKERS)

    train(train_dataloader, logger, in_channels = 3, optimizer = config.OPTIMIZER)

if __name__ == "__main__":
    main()