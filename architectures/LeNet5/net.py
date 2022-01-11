import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeNet5"]


class LeNet5(nn.Module):
    def __init__(self, is_colored: bool = False, num_classes: int = 10) -> None:
        super(LeNet5, self).__init__()
        in_channels = 3 if is_colored else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc_layers = nn.Sequential(
            nn.Linear(400, 120),
            # nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x_flatted = x.view(x.shape[0], -1)
        out = self.fc_layers(x_flatted)
        return out


def main():
    model = LeNet5()
    print(model)


if __name__ == "__main__":
    main()
