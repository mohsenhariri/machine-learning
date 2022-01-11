from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from data_augmentation.utility import show2, show3
import torch.nn.functional as F


img = Image.open("./data/custom_cat_dog/cat1.jpg")
img = ImageOps.grayscale(img)

# img.show()
transform = T.ToTensor()

im_tensor = transform(img)
im_tensor = im_tensor.reshape([1, 1, 800, 640])
# print(im_tensor.size())

kernel = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
emboss_kernel = [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]
sobel_kernel = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]


filter1 = torch.tensor([emboss_kernel])
filter2 = torch.tensor([sobel_kernel])


filter1 = filter1.view(1, 1, 3, 3)
filter2 = filter2.view(1, 1, 3, 3)


out1 = F.conv2d(input=im_tensor, weight=filter1)
out2 = F.conv2d(input=im_tensor, weight=filter2)

im_cov1 = out1[0, 0, :, :]
im_cov2 = out2[0, 0, :, :]

# plt.imshow(ph, cmap='gray', vmin=0, vmax=255)
# plt.imshow(ph1, cmap="gray")
# plt.imshow(ph2, cmap="gray")

f, ax = plt.subplots(1, 2)
ax[0].imshow(im_cov1, cmap="gray")
ax[1].imshow(im_cov2, cmap="gray")
plt.show()
