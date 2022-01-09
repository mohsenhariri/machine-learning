import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor


class ConvBlock(nn.Module):  # conv + batch norm + relu
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, padding) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(ConvBlock(in_channels, out_channels=ch1x1, kernel_size=1, stride=1, padding=0))

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_channels=ch3x3red, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=0),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out_channels=ch5x5red, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=0),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, out_channels=pool_proj, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.concat([x1, x2, x3, x4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)

        x = self.conv(x)
        x = self.act(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super(GoogLeNet, self).__init__()

        # self.conv1 =
