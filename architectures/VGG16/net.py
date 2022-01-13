import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VGG16"]


class VGG16(nn.Module):
    def __init__(self, init_weights: bool = False, num_classes: int = 1000) -> None:
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x

    def _initialize_weights(self) -> None:
        pass
