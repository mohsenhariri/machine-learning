import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        # nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.xavier_normal_(self.fc3.weight)

        self._initialization()

    def forward(self, x) -> torch.Tensor:
        return x

    def _initialization(self) -> None:
        for layer in self.layermodules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                # nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.constant_(layer.bias, 0)
