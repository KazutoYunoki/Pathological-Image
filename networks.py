from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import csv
from pathlib import Path


class vgg_fcn(nn.Module):

    def __init__(self):
        super(vgg_fcn, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, False),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, False),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, False),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )

    def forward(self, input):
        output = self.features(input)
        output = self.avgpool(output)
        output = output.view(1, -1)
        output = self.classifier(output)
        return output


class FCNs(nn.Module):
    """
    Full Convolutional network(FCN8s)の実装
    独自にカスタマイズしてあるネットワークモデル
    """

    def __init__(self):
        super(FCNs, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        # 以下custom up8
        self.up8 = nn.Sequential(
            nn.Conv2d(4096, 512, 1),
            nn.ConvTranspose2d(512, 512, 4, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

        # up9
        self.up9 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        # up10
        self.up10 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.up11 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.fc6(output)
        output = self.fc7(output)
        output = self.up8(output)
        output = self.up9(output)
        output = self.up10(output)
        output = self.up11(output)
        output = self.last(output)
        output = output.view(input.shape[0], 3, 32*32)

        return output


if __name__ == "__main__":

    net = FCNs()
    print(net)

    """
    with open(output_dir + '/color.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(8):
            writer.writerow(['予測結果'])
            writer.writerows(output[i])
    
    print(net)
    conv1 = nn.Conv2d(4096, 512, 1)
    conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, bias=False)
    conv3 = nn.Conv2d(512, 256, 1)
    conv4 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
    conv5 = nn.Conv2d(256, 128, 1)
    conv6 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)
    conv7 = nn.Conv2d(128, 64, 1)
    conv8 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False)
    conv9 = nn.Conv2d(64, 3, 1)

    input = torch.randn(1, 4096, 1, 1)
    output = conv1(input)
    print(output.shape)
    output = conv2(output)
    print(output.shape)
    output = conv3(output)
    print(output.shape)
    output = conv4(output)
    print(output.shape)
    output = conv5(output)
    print(output.shape)
    output = conv6(output)
    print(output.shape)
    output = conv7(output)
    print(output.shape)
    output = conv8(output)
    print(output.shape)

    output = conv9(output)
    print(output.shape)
    output = output.view(1, 3, 32*32)
    print(output.shape)
    """
