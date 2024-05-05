import torch
from torch import nn
import torchvision
import math
from torch.nn.utils import spectral_norm as SpectralNorm

## As found in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution ##
## With some changes (residual scaling, adding GELU, attention, disabling batch normalisation, etc.) ##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, batch_norm = False, activation = None, spectral_norm = False):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        stride: stride
        batch_norm: include a BN layer?
        activation: Type of activation; None if none
        spectral_norm: use spectral normalisation?
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {"prelu", "leakyrelu", "tanh", "gelu"}

        # A container that will hold the layers in this convolutional block
        layers = list()

 

        # A batch normalisation (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features = out_channels))

        # An activation layer, if wanted
        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "gelu":
            layers.append(nn.GELU())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        input: input images, a tensor of size (N, in_channels, w, h)
        returns: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output

class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size = 3, n_channels = 64, scaling_factor = 2, activation = "prelu"):
        """
        kernel_size: kernel size of the convolution
        n_channels: number of input and output channels
        scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels = n_channels, out_channels = n_channels * (scaling_factor ** 2),
                              kernel_size = kernel_size, padding = kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = scaling_factor)
        
        activation = activation.lower()
        assert activation in {"prelu", "leakyrelu", "tanh", "gelu"}

        # Activation layer
        if activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()

    def forward(self, input):
        """
        Forward propagation.

        input: input images, a tensor of size (N, n_channels, w, h)
        returns: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.activation(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output

class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size = 3, n_channels = 64, activation = "PReLU", batch_norm = True, resid_scale_factor = None):
        """
        kernel_size: kernel size
        n_channels: number of input and output channels (same because the input must be added to the output)
        activation: activation function
        scale_factor: factor for scaling residuals
        """
        super(ResidualBlock, self).__init__()

        self.resid_scale_factor = resid_scale_factor

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size,
                                              batch_norm = batch_norm, activation = activation, spectral_norm = False)

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size,
                                              batch_norm = batch_norm, activation = None, spectral_norm = False)
        
    def forward(self, input):
        """
        Forward propagation.

        input: input images, a tensor of size (N, n_channels, w, h)
        returns: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)

        if (isinstance(self.resid_scale_factor, int) == True) or (isinstance(self.resid_scale_factor, float) == True):
            output = output.mul(self.resid_scale_factor)

        output = output + residual  # (N, n_channels, w, h)

        return output

class SRResNet(nn.Module):
    """
    SRResNet with attention and activation customisability
    """

    def __init__(self, large_kernel_size = 9, small_kernel_size = 3, n_channels = 64, n_blocks = 16, scaling_factor = 4, activation = "PReLU", enable_standard_bn = True, resid_scale_factor = None, self_attention = False):
        """
        large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        n_blocks: number of residual blocks
        scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        self_attention: include self-attention?
        """
        super(SRResNet, self).__init__()

        self.resid_scale_factor = resid_scale_factor
        self.self_attention = self_attention

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels = 3, out_channels = n_channels, kernel_size = large_kernel_size,
                                              batch_norm = False, activation = activation, spectral_norm = False)

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size = small_kernel_size, n_channels = n_channels, activation = activation, batch_norm = enable_standard_bn, resid_scale_factor = resid_scale_factor) for i in range(n_blocks)])

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels = n_channels, out_channels = n_channels, kernel_size = small_kernel_size,
                                              batch_norm = enable_standard_bn, activation = None, spectral_norm = False)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size = small_kernel_size, n_channels = n_channels, scaling_factor = 2, activation = activation) for i
              in range(n_subpixel_convolution_blocks)])

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels = n_channels, out_channels = 3, kernel_size = large_kernel_size,
                                              batch_norm = False, activation = "Tanh")
        if self.self_attention == True:
            self.attention = Self_Attention(n_channels)

    def forward(self, lr_imgs):
        """
        Forward propagation

        lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        returns: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)

        if (isinstance(self.resid_scale_factor, int) == True) or (isinstance(self.resid_scale_factor, float) == True):
            output = output.mul(self.resid_scale_factor)

        output = output + residual  # (N, n_channels, w, h)
        
        if self.self_attention == True:
            output = self.attention(output) # (N, n_channels, w, h)

        output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

        return sr_imgs

class Generator(nn.Module):
    """
    The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
    """

    def __init__(self, large_kernel_size = 9, small_kernel_size = 3, n_channels = 64, n_blocks = 16, scaling_factor = 4, activation = "PReLU", enable_standard_bn = True, resid_scale_factor = None, self_attention = False):
        """
        large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        n_blocks: number of residual blocks
        scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        self_attention: include self-attention?
        """
        super(Generator, self).__init__()

        # The generator is simply SRResNet, as above
        self.net = SRResNet(large_kernel_size = large_kernel_size, small_kernel_size = small_kernel_size, n_channels = n_channels, n_blocks = n_blocks, scaling_factor = scaling_factor, activation = activation, enable_standard_bn = enable_standard_bn, resid_scale_factor = resid_scale_factor, self_attention = self_attention)

    def initialise_with_srresnet(self, srresnet_checkpoint):
        """
        Initialise with weights from a trained SRResNet.

        srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint, map_location = device)["model"]
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs):
        """
        Forward prop.

        lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        returns: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs

class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper, with self-attention
    """

    def __init__(self, kernel_size = 3, n_channels = 64, n_blocks = 8, fc_size = 1024, self_attention = False, spectral_norm = False):
        """
        kernel_size: kernel size in all convolutional blocks
        n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        n_blocks: number of convolutional blocks
        fc_size: size of the first fully connected layer
        self_attention: include self-attention?
        spectral_norm: use spectral normalisation?
        """
        super(Discriminator, self).__init__()

        self.self_attention = self_attention

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalisation
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels

            conv_blocks.append(
                ConvolutionalBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                                    stride = 1 if i % 2 == 0 else 2, batch_norm = (i != 0) and (spectral_norm is False), activation = "LeakyReLU", spectral_norm = spectral_norm))

            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(fc_size, 1)

        if spectral_norm == True:
            self.fc1 = SpectralNorm(self.fc1)
            self.fc2 = SpectralNorm(self.fc2)

        if self.self_attention == True:
            self.attention = Self_Attention(out_channels)

    def forward(self, imgs):
        """
        Forward propagation

        imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        returns: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        if self.self_attention == True:
            output = self.attention(output)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit
    
class Self_Attention(nn.Module):
    """ 
    Self-Attention layer as in https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8 , kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8 , kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim = -1) #
    
    def forward(self, x):
        """
        x: input feature maps (B X C X W X H)
        out: self attention value + input feature 
        attention: B X N X N (N is Width * Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out
