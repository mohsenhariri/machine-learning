from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

image = Image.open("./data/custom_cat_dog/cat1.jpg")
transform = T.ToTensor()

img = transform(image)


img_p = F.max_pool2d(input=img, kernel_size=3, return_indices=True)

img_maxPooled, indices_arg_max = img_p

# print(indices_arg_max.size(), img_maxPooled.size())
# exit()
# for i in range(20):
# img_maxPooled = F.max_pool2d(input=img_maxPooled, kernel_size=3, stride=1, padding=1, ceil_mode=True)


img_unPooled = F.max_unpool2d(input=img_maxPooled, indices=indices_arg_max, kernel_size=4)
# F.max_unpool2d()
# nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),


# plt.imshow(img_maxPooled.permute(1, 2, 0))
plt.imshow(img_unPooled.permute(1, 2, 0))

plt.show()
