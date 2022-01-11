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


# t23 = torch.tensor([[1, 0], [0, 1]]).reshape([1, 1, 2, 2])
# t23 = torch.tensor([[1, 0], [0, 1]]).reshape([1, 1, 2, 2])

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
initial_value = torch.tensor(kernel)


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
        # nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.xavier_normal_(self.fc3.weight)
        with torch.no_grad():
            self.conv1.weight.copy_(initial_value)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        return out

    def help(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
                # nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.weight, 0)
                # nn.init.(m.weight, 0)
                # m.weight = nn.parameter(t23)

                # if m.bias is not None:
                #     print("ok")
                #     nn.init.constant_(m.bias, 0)

                # print(m.weight, m.bias)
                # print(m.weight.size())
        pass


model = NN()


model.help()
#

# print(t23.size())
for batch_ndx, (x, y) in enumerate(train_loader):
    img_grid = torchvision.utils.make_grid(x)
    writer.add_image(tag=f"Original- batch_iter: {batch_ndx}", img_tensor=img_grid)

    output_conv = model(x)
    img_grid = torchvision.utils.make_grid(output_conv)
    writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
    writer.close()

    exit()
