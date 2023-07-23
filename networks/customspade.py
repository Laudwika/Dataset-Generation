from __future__ import print_function
from binascii import b2a_base64
from netrc import netrc
from re import A
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import torchvision.models as models
import random
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from SPADE.models.networks.generator import SPADEGenerator
from SPADE.models.networks.discriminator import MultiscaleDiscriminator
from SPADE.options.train_options import TrainOptions
import torch.distributed as dist
import torchvision.models as models
from torch.autograd import Variable

num_classes = 3
opt = TrainOptions().parse()
opt.semantic_nc = num_classes  # The number of classes in your segmentation map
opt.label_nc = num_classes
opt.ngf = 64  # The number of generator filters in the first layer
opt.ndf = 64

class CustomSPADE(nn.Module):
    def __init__(self, num_classes):
        super(CustomSPADE, self).__init__()
        self.encoder_skin = Encoder(input_channels=3, output_channels=256)
        self.encoder_hair = Encoder(input_channels=3, output_channels=256)
        self.encoder_left_eye = Encoder(input_channels=3, output_channels=256)
        self.encoder_right_eye = Encoder(input_channels=3, output_channels=256)
        self.encoder_mouth = Encoder(input_channels=3, output_channels=256)
        self.encoder_seg = Encoder(input_channels=3, output_channels=256)
        self.fusion_layer = FusionLayer(num_features=256)
        self.mouth_fusion_layer = MouthFusionLayer(num_features=256)
        self.blending_layer = BlendingLayer()
        self.mouth_transform = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.spade_generator = SPADEGenerator(opt)
        self.discriminator = MultiscaleDiscriminator(opt,6)

    def forward(self, seg_map, skin, hair, left_eye, right_eye, mouth):
        skin_features = self.encoder_skin(skin)
        hair_features = self.encoder_hair(hair)
        left_eye_features = self.encoder_left_eye(left_eye)
        right_eye_features = self.encoder_right_eye(right_eye)
        mouth_features = self.encoder_mouth(mouth)
        mouth_features = self.mouth_transform(mouth_features)
        seg_features = self.encoder_seg(seg_map)

        fused_features = self.fusion_layer(seg_map, seg_features, skin_features, hair_features, left_eye_features, right_eye_features)
        
        fused_mouth_features = self.mouth_fusion_layer(seg_features, mouth_features)
        fused_features = self.blending_layer(fused_features, fused_mouth_features)

        output = self.spade_generator(fused_features, seg_map)
        return output




class Encoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return x

class BlendingLayer(nn.Module):
    def __init__(self):
        super(BlendingLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, 3, 1, 1))

    def forward(self, x1, x2):
        weights = torch.sigmoid(self.weights)
        blended = x1 * weights + x2 * (1 - weights)
        return blended

class MouthFusionLayer(nn.Module):
    def __init__(self, num_features=64):
        super(MouthFusionLayer, self).__init__()
        self.conv = nn.Conv2d(num_features * 2, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, seg_features, mouth_features):
        fused_features = torch.cat([seg_features, mouth_features], dim=1)
        fused_features = self.conv(fused_features)
        return fused_features

class FusionLayer(nn.Module):
    def __init__(self, num_features=64):
        super(FusionLayer, self).__init__()
    #     self.conv = nn.Conv2d(num_features * 6, 3, kernel_size=1, stride=1, padding=0)

    # def forward(self, seg_map,seg_features, skin_features, hair_features, left_eye_features, right_eye_features, mouth_features):
    #     fused_features = torch.cat([seg_features, skin_features, hair_features, left_eye_features, right_eye_features, mouth_features], dim=1)
    #     fused_features = self.conv(fused_features)
    #     return fused_features
        self.conv = nn.Conv2d(num_features * 5, 3, kernel_size=1, stride=1, padding=0)
        self.gate_conv = nn.Conv2d(num_features * 5, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg_map, seg_features, skin_features, hair_features, left_eye_features, right_eye_features):
        concatenated_features = torch.cat([seg_features, skin_features, hair_features, left_eye_features, right_eye_features], dim=1)
        
        combined_features = self.conv(concatenated_features)
        gate_weights = self.sigmoid(self.gate_conv(concatenated_features))
        
        gated_features = combined_features * gate_weights
        return gated_features
