from typing import Tuple
import torch
import torch.nn as nn

__all__ = ["AlexNet"]


def out_dimension(w, kernel_size, stride_size=1, padding=0):
    return int((w - kernel_size + 2 * padding) / stride_size) + 1


class AlexNet(nn.Module):
    def __init__(self, is_colored: bool = True, input_size: Tuple[int, int] = (224, 224), num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()
        in_channels = 3 if is_colored else 1
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        h, w = input_size

        h = out_dimension(h, 11, 4, 2)  # Layer C1: Convolution Layer (96, 11×11)
        h = out_dimension(h, 3, 2)  # Layer S2: Max Pooling Layer (3×3)
        h = out_dimension(h, 5, 1, 2)  # Layer C3: Convolution Layer (256, 5×5)
        h = out_dimension(h, 3, 2)  # Layer S4: Max Pooling Layer (3×3)
        h = out_dimension(h, 3, 1, 1)  # Layer C5: Convolution Layer (384, 3×3)
        h = out_dimension(h, 3, 1, 1)  # Layer C6: Convolution Layer (384, 3×3)
        h = out_dimension(h, 3, 1, 1)  # Layer C7: Convolution Layer (256, 3×3)
        h = out_dimension(h, 3, 2)  # Layer S8: Max Pooling Layer (3×3)
        w = h
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=256 * h * w,
                out_features=4096,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=4096,
                out_features=4096,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=4096,
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    from torchinfo import summary

    model = AlexNet()
    summary(model=model, input_size=(100, 3, 224, 224))


if __name__ == "__main__":
    main()
