from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from utility import show2, show3

img = Image.open("./data_augmentation/data/cat1.jpg")
transformF = T.ToTensor()

im_tensor = transformF(img)

convLayerF = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=2, stride=2)
normLayerF = nn.BatchNorm2d(num_features=1, eps=0.0001, momentum=0.1)
maxPooingLayerF = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))


im_pooled = maxPooingLayerF(im_tensor)
im_batch_normed = normLayerF(im_tensor)
# show2(im_tensor[0], im_pooled[0])
show3(im_tensor[0], im_pooled[0], im_batch_normed[0])
