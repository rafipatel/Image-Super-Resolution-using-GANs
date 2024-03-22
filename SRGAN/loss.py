import torch.nn as nn
from torchvision.models import vgg19 # They used vgg19 specifically
# import config

# The layer that they used the output from VGG is a specific one,
# in paper its --> phi_5,4, which is I think 5th conv layer before maxpooling but after activation, something like that


class VGGLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE) #we get 36 list of features, which is calculated from above phi_5,4
    #.eval() so we won't update the weights, also may be they used dropout so its called here
    self.loss = nn.MSELoss()


    for param in self.vgg.parameters():
      param.required_grad = False # Not to update it


  def forward(self,input,target): #input -->Upscaled low res image , target --> original high quality image
     vgg_input_features = self.vgg(input)
     vgg_target_features = self.vgg(target)

     return self.loss(vgg_input_features, vgg_target_features)

