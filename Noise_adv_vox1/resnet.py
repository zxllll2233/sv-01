import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input,constant):
        ctx.alpha=constant
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output,None

def grad_reverse(x,constant):
    return GRLayer.apply(x,constant)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class resnet_18(nn.Module):
    def __init__(self,outchannels):
        self.outchannels = outchannels
        super(resnet_18,self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resnet18 = nn.Sequential( self.b1, self.b2, self.b3, self.b4, self.b5,
                                      nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Flatten(), nn.Linear(512, self.outchannels))

    def forward(self, x,constant):
        x = x.view(x.size(0), 1, 512, 202)
        x = grad_reverse(x, constant)
        x = self.resnet18(x)

        return x #(b,512,outchannels)

class embedding_discriminator(nn.Module):
    def __init__(self):
        super(embedding_discriminator,self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)
        )

    def forward(self,x,constant):
        x=grad_reverse(x,constant)
        x=self.discriminator(x)

        return x