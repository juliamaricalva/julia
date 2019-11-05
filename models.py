import torch
import torchvision
import torch.nn as nn



class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained='imagenet')
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,bias=False)
        self.convt4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,padding=1, bias=False)
        self.convt5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2,padding=1, bias=False)
        self.conv1 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        x = self.resnet18(img)
        x = self.relu(self.convt1(x))
        x = self.relu(self.convt2(x))
        x = self.relu(self.convt3(x))
        x = self.relu(self.convt4(x))
        x = self.relu(self.convt5(x))
        x = self.conv1(x)

        return x





