import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      discriminator=False,
      # to specify if its in discriminator or not
      use_act = True, # will use an activation or not
      use_bn = True, # Use batchnorm or not
      **kwargs,
      ):

    super().__init__()
    self.use_act = use_act
    self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias = not use_bn )  # Bcoz Bias can be false when we are not using batchnorm
    self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
    self.act = (     #activatiom
        nn.LeakyReLU(0.2, inplace=True)
        if discriminator
        else nn.PReLU(num_parameters=out_channels)  #We can train Prelu to spcify the slope (to read more) , that means each of the out_channels will have a seperate slope. Set as per paper
                    )

  def forward(self,x):
    return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))



class UpsampleBlock(nn.Module):
  def __init__(self, in_c, scale_factor):  # in_c --> in channels
    super().__init__()
    self.conv = nn.Conv2d(in_c, in_c * scale_factor **2, 3,1,1)  # The reason for increasing channels here (* scalefactor**2) instead fo bilinear upsample (i.e increase height and width), because we are using pixelshuffle
    self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
    self.act = nn.PReLU(num_parameters = in_c)  # in_c because after the pixel shifter it will still be in c number of channels


  def forward(self,x):
    return self.act(self.ps(self.conv(x))) #activation of pixel shuffle of conv

class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.block1 = ConvBlock(
        in_channels,
        in_channels, #We won't modify number of channels as per architecture in paper
        kernel_size = 3,
        stride = 1,
        padding = 1
        )

    self.block2 = ConvBlock(
        in_channels,
        in_channels, #We won't modify number of channels as per architecture in paper
        kernel_size = 3,
        stride = 1,
        padding = 1,
        use_act=False,
        )

  def forward (self,x):
    out = self.block1(x)
    out = self.block2(out)

    return out + x # "+ x" because we need skip connection that was inputted to the block


class Generator(nn.Module):
  def __init__(self, in_channels=3, num_channels=64, num_blocks=16): # Values as per architecture in paper
     super().__init__()
     self.initial = ConvBlock(in_channels, num_channels, kernel_size =9, stride = 1, padding =4, use_bn = False) # False, since no batch norm in the beginning in paper)
     self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)]) #List comprehension to create all of 16 block, and asterisk to unwrap that list of residual block and turning it into Sequential
     self.convblock = ConvBlock(num_channels, num_channels, kernel_size = 3, stride =1 , padding =1, use_act=False)
     self.upsamples = nn.Sequential(UpsampleBlock(num_channels, scale_factor =2),UpsampleBlock(num_channels, scale_factor =2))
     self.final = ConvBlock(num_channels,in_channels, kernel_size = 9 , stride=1, padding=4 )

  def forward(self,x):
    initial = self.initial(x)
    x = self.residuals(initial) #initial specidied, bcoz we want skip conncetion after the first conv layer
    x = self.convblock(x) + initial # elementwise sum from paper archi
    x = self.upsamples(x)
    return torch.tanh(self.final(x)) #not clear they used tanh, (but they normalised between 1 & -1, so it make sense to use tanh)

class Discriminator(nn.Module):
  def __init__(self, in_channels = 3, features = [64,64,128,128,256,256,512,512] ):
     #mentioned in architecture
      super().__init__()

      blocks = []

      for idx, feature in enumerate(features):
        blocks.append(
            ConvBlock(
                in_channels,
                feature,
                kernel_size=3,
                stride=1 + idx % 2, # stride is 1 in the beginning or architecture then 2 then 1 then 2 and so on
                # when index is 0 its 1 + 1 % 2 = 1+0 =0, similarly when idx is 1, 1+1%2 =2 , since anything modulus(remainder) by 2 is either 0 or 1
                padding=1,
                discriminator=True, # since we have set it to false in ConV block
                use_act=True,
                use_bn = False if idx==0 else True # Since batch norm is for all blocks except the first one in paper
                )
                )
        in_channels = feature

        self.blocks = nn.Sequential(*blocks) #unwrapping the blocks list

        # classifier turn now , flow from paper = Dense (1024) --> LeakyReLU-->Dense-->Sigmoid

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)), # its 6x6 when its outputted, for instance, if we keep on dividing by stride of 2
            # AdaptiveAvgPool2d won't do anythong if ots 96x96, but will make sure it runs when its bigger value like 128,192
            nn.Flatten(),
            nn.Linear(512*6*6, 1024), # 512 is the number of channels in output
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,1)
            )


  def forward(self,x):
      x = self.blocks(x)
      return self.classifier(x)

      #They used sigmoid inpaper, we will specify BC with logits which includes sigmoid anyways




def test():

  low_resolution =100 # High resolution is 96x96 when we divide by 4 lower resolution is 24x24. Thats what we are going to run this with
  # so when we upsample 24x24 --> we get 96x96 from the generator, then we run throgh discriminator where we get only one single output, where batch size is 5

  with torch.autocast(device_type="cuda"):
    x = torch.randn((5, 3, low_resolution, low_resolution))  #batch size, number of channels, Image height, Image width
    gen = Generator()
    gen_out=gen(x)
    disc = Discriminator()
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

if __name__ == "__main__":
  test()

