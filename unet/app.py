import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubuleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubuleConv, self).__init__()
        self.conv = nn.Sequential()



class 