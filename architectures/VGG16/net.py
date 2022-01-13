from typing import Tuple
import torch
import torch.nn as nn

__all__ = ["VGG16"]


class VGG16(nn.Module):
    def __init__(self, init_weights: bool = False, input_size: Tuple[int, int] = (224, 224), dropout: float = 0.5, num_classes: int = 1000) -> None:
        super(VGG16, self).__init__()

        self.feature1 = self.conv2d_2layer(3, 64)
        self.feature2 = self.conv2d_2layer(64, 128)

        self.feature3 = self.conv2d_3layer(128, 256)
        self.feature4 = self.conv2d_3layer(256, 512)
        self.feature5 = self.conv2d_3layer(512, 512)

        h = int(input_size[0] / 2 ** 5)
        w = int(input_size[1] / 2 ** 5)

        self.avgpool = nn.AdaptiveAvgPool2d((h, w))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * h * w, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        z = self.avgpool(f5)
        # a = z.view(z.size(0), -1)
        a = torch.flatten(z, 1)
        out = self.classifier(a)
        return out

    def _initialize_weights(self) -> None:
        pass

    def conv2d_2layer(self, in_channels, out_channels) -> None:

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def conv2d_3layer(self, in_channels, out_channels) -> None:

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


def main():
    model = VGG16()
    print(model)


if __name__ == "__main__":
    main()

"""

        self.conv1 = conv2d(in_channels=3, out_channels=64)
        self.conv2 = conv2d(in_channels=64, out_channels=64)

        self.conv3 = conv2d(in_channels=64, out_channels=128)
        self.conv4 = conv2d(in_channels=128, out_channels=128)

        self.conv5 = conv2d(in_channels=128, out_channels=256)
        self.conv2 = conv2d(in_channels=256, out_channels=256)
        self.conv2 = conv2d(in_channels=256, out_channels=256)

        self.conv2 = conv2d(in_channels=256, out_channels=512)
        self.conv2 = conv2d(in_channels=512, out_channels=512)
        self.conv2 = conv2d(in_channels=512, out_channels=512)

        self.conv2 = conv2d(in_channels=512, out_channels=512)
        self.conv2 = conv2d(in_channels=512, out_channels=512)
        self.conv2 = conv2d(in_channels=512, out_channels=512)


"""
