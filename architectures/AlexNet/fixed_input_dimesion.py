import torch
import torch.nn as nn

__all__ = ["AlexNet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()
        self.features = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=11,
                    stride=4,
                    padding=2,
                ),
                nn.ReLU(inplace=True),  # next value will be in place of old value
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                ),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=192,
                    kernel_size=5,
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
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=256 * 6 * 6,
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
    model = AlexNet()

    print(model)


if __name__ == "__main__":
    main()
