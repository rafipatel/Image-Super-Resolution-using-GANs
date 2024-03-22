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
import wandb
# from google.colab import files


torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):

    wandb.init(project='SRGAN', entity='rafi-patel')
    
    loop = tqdm(loader, leave=True)
    
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        ) #one sided label smoothing (extra), rest is as per paper

        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        print('='*50)
        print(disc_loss_fake)
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
        # gen_loss =  l2_loss
        print(gen_loss)
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        a = {'gen_loss(includingVGGloss)': gen_loss.item(),
                   'loss_for_vgg' : loss_for_vgg,
                    'disc_loss': loss_disc.item(),
                      'mse_loss': l2_loss.item(), 
                      'adversarial_loss': adversarial_loss.item()}
        print(a)
        
        wandb.log({'gen_loss(includingVGGloss)': gen_loss.item(),
                   'loss_for_vgg' : loss_for_vgg,
                    'disc_loss': loss_disc.item(),
                      'mse_loss': l2_loss.item(), 
                      'adversarial_loss': adversarial_loss.item()})

        # if idx % 200 == 0:

        #     low_res_valid_folder = "/root/tensorflow_datasets/downloads/extracted/ZIP.data.visi.ee.ethz.ch_cvl_DIV2_DIV2_vali_LRpQpdHEuI3k6NMA2PsrkExS_pOyspikiZaXdg18u21VM.zip/DIV2K_valid_LR_bicubic/X4"
        #     plot_examples(low_res_valid_folder, gen)


def main():
    # dataset = MyImageFolder(root_dir="new_data/")

    root_dir = "G:\My Drive\GAN"

    dataset = SuperResolutionDataset(root_dir=root_dir)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )


    # loader = DataLoader(dataset, batch_size=128) #, num_workers=8)


    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    
    
    if config.LOAD_MODEL:

        # load_checkpoint(
        #     CHECKPOINT_GEN,
        #     gen,
        #     opt_gen,
        #     LEARNING_RATE,
        # )
        # load_checkpoint(
        #    CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        # )


# def load_checkpoint(checkpoint_file, model, optimizer, lr):
        print("=> Loading checkpoint")
        checkpoint = torch.load("/content/gen_100_epochs.tar")
        gen.load_state_dict(checkpoint["state_dict"])
        opt_gen.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in opt_gen.param_groups:
        param_group["lr"] = config.LEARNING_RATE

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)
        print(epoch)
        if epoch%100 == 0:
          if config.SAVE_MODEL:
              # save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
              # save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

              # def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
              print("=> Saving checkpoint Generator")
              checkpoint = {
                  "state_dict": gen.state_dict(),
                  "optimizer": opt_gen.state_dict(),
              }

              torch.save(checkpoint, '/content/gen_1000_epochs.tar')
              # files.download('/content/gen_1000_epochs.tar')
              # print("Downloading checkpoint Generator")


              print("=> Saving checkpoint Discriminator")

              checkpoints = {
                  "state_dict": disc.state_dict(),
                  "optimizer": opt_disc.state_dict(),
              }
              torch.save(checkpoints, '/content/disc_1000_epochs.tar')
              # print("Downloading checkpoint Discriminator")
              # files.download('/content/disc_1000_epochs.tar')


     



        print(epoch, "Completed")
        wandb.finish()



if __name__ == "__main__":
    main()