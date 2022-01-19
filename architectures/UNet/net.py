from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# from arguments import hp


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2) -> None:
        super(Unet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, feature in enumerate([64, 128, 256, 512]):
            self.downs.add_module(f"up{i+1}", DoubleConv(in_channels=in_channels, out_channels=feature))
            # self.downs.add_module(f"pool{i+1}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        self.bottleneck = DoubleConv(in_channels=512, out_channels=1024)

        in_channels = 1024
        for i, feature in enumerate([512, 256, 128, 64]):
            self.ups.add_module(f"up{i}", DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

            # in_channels = feature

        self.out = nn.Conv2d(in_channels=feature, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.downs:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        out = self.out(x)

        return out


if __name__ == "__main__":
    model = Unet()
    print(model)


