import numpy as np
import torch.nn as nn
import torchvision
import torch
import math


class PSP(nn.Module):

    def __init__(self, in_channels):

        super(PSP, self).__init__()

        out_channels = int(in_channels / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Dropout2d(0.1, False))

        for i in self.modules():

            if isinstance(i, nn.Conv2d):
                n = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()

    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        x1 = nn.functional.adaptive_avg_pool2d(x, 1)
        x1 = self.conv1(x1)
        x1 = nn.functional.upsample_bilinear(x1, size=fcn_features_spatial_dim)
        x2 = nn.functional.adaptive_avg_pool2d(x, 2)
        x2 = self.conv2(x2)
        x2 = nn.functional.upsample_bilinear(x2, size=fcn_features_spatial_dim)
        x3 = nn.functional.adaptive_avg_pool2d(x, 3)
        x3 = self.conv3(x3)
        x3 = nn.functional.upsample_bilinear(x3, size=fcn_features_spatial_dim)
        x4 = nn.functional.adaptive_avg_pool2d(x, 6)
        x4 = self.conv4(x4)
        x4 = nn.functional.upsample_bilinear(x4, size=fcn_features_spatial_dim)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.fusion_bottleneck(x)

        return x


class Resnet50(nn.Module):

    def __init__(self, args, num_classes=9):
        super(Resnet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.psp = PSP(resnet50.inplanes)
        resnet50.fc = nn.Conv2d(resnet50.inplanes // 4, num_classes, 1)
        self.resnet50 = resnet50

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.psp(x)
        x = self.resnet50.fc(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x