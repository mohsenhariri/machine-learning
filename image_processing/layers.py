from PIL import Image,ImageOps
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from data_augmentation.utility import show2, show3
import torch.nn.functional as F

img = Image.open("./data/custom_cat_dog/cat1.jpg")
img = ImageOps.grayscale(img)

# img.show()
transformF = T.ToTensor()

im_tensor = transformF(img)
im_tensor = im_tensor.reshape([1, 1, 800, 640])
print(im_tensor.size())

# convLayerF = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1), bias=False)
# normLayerF = nn.BatchNorm2d(num_features=1, eps=0.0001, momentum=0.1)
# maxPooingLayerF = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))


# im_pooled = maxPooingLayerF(im_tensor)
weights = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
    ]
)

weights = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
print(weights.size())
weights = weights.view(2, 1, 3, 3)


out = F.conv2d(input=im_tensor, weight=weights)



print(out.size())


exit()
with torch.no_grad():

    im_convoled = convLayerF(im_tensor[0], weights)
# im_batch_normed = normLayerF(im_tensor)
# show2(im_tensor[0], im_pooled[0])
# show3(im_tensor[0], im_pooled[0], im_batch_normed[0])
