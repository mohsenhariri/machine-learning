import torch.nn as nn
from collections import OrderedDict


# model = nn.Sequential()


layer1conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
out1relu = nn.ReLU()
layer2pool = nn.MaxPool2d(kernel_size=2)
layer3conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
out3relu = nn.ReLU()
layer4pool = nn.MaxPool2d(kernel_size=2)
layer5fc = nn.Linear(32 * 5 * 5, 10)

model = nn.Sequential(
    OrderedDict(
        [
            ("layer1conv", layer1conv),
            ("out1relu", out1relu),
            ("layer2pool", layer2pool),
            ("layer3conv", layer3conv),
            ("out3relu", out3relu),
            ("layer4pool", layer4pool),
            ("layer5fc", layer5fc),
        ]
    )
)


print(model)
