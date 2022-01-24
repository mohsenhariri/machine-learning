from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet"]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, norm: bool = True, plain_shortcut: bool = True) -> None:
        super(Bottleneck, self).__init__()

        self.weigh_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        if not plain_shortcut:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        out_weight_layers = self.weigh_layers(x)
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        out = F.relu(out_weight_layers + x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, norm: bool = True, plain_shortcut: bool = True) -> None:
        super(BasicBlock, self).__init__()

        self.weigh_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        if not plain_shortcut:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        out_weight_layers = self.weigh_layers(x)
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        out = F.relu(out_weight_layers + x)
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock: nn.Module, layers: List[int], num_class: int = 1000) -> None:
        super(ResNet, self).__init__()

        self.base_layer = nn.Sequential(
            nn.Conv2d(1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        res = ResBlock(in_channels=3, channels=10)
        self.fm = nn.Sequential(res)

        self.layer1 = self._make_layer(ResBlock, 64, layers[0])
        self.layer2 = self._make_layer(ResBlock, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(ResBlock, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(ResBlock, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.base_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, channels, num_layers):
        layers = []

        for _ in range(1, num_layers):
            layers.append(block())

        return nn.Sequential(*layers)



# model = ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
layers = [2, 2, 2, 2]

model = ResNet(ResBlock=BasicBlock, layers=layers)

print(model)


# return


# def ResNet50(img_channel=3, num_classes=1000):
#     return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


reses = [18, 34, 50, 101, 152]

layers = [2, 2, 2, 2]
layers = [3, 4, 6, 3]
layers = [3, 4, 6, 3]





def main():
    from torchinfo import summary

    model = ResNet()
    # summary(model=model, input_size=(100, 3, 224, 224)) ???


if __name__ == "__main__":
    main()