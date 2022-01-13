from os import environ, path
import torch.nn as nn
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# from data.mnist import train_loader,test_loader
# from data.cifar import train_loader, test_loader
from data.custom_cat_dog import train_loader

j = 0
while True:
    j += 1
    file_exists = path.exists(f"./image_processing/runs/{j}")
    if not file_exists:
        break
writer = SummaryWriter(log_dir=f"./image_processing/runs/{j}")


kernel = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
emboss_kernel = [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]
sobel_kernel = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]


initial_value = torch.tensor(kernel)


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

        with torch.no_grad():
            self.conv1.weight.copy_(initial_value)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        return out


model = NN()


for batch_ndx, (x, y) in enumerate(train_loader):
    img_grid = torchvision.utils.make_grid(x)
    writer.add_image(tag=f"Original- batch_iter: {batch_ndx}", img_tensor=img_grid)

    output_conv = model(x)
    print(output_conv.size())

    img_grid0 = torchvision.utils.make_grid(output_conv[0:, 0, :, :])
    img_grid1 = torchvision.utils.make_grid(output_conv[0:, 1, :, :])
    img_grid2 = torchvision.utils.make_grid(output_conv[0:, 2, :, :])

    writer.add_image(tag=f"layer0: {batch_ndx}", img_tensor=img_grid0)
    writer.add_image(tag=f"layer1: {batch_ndx}", img_tensor=img_grid1)
    writer.add_image(tag=f"layer2: {batch_ndx}", img_tensor=img_grid2)

    writer.close()

    exit()
