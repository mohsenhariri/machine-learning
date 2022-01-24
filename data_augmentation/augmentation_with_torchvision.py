from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from data_augmentation.utility import show2, show3, show4

# from torchvision.transforms.transforms import ToTensor


img = Image.open("./data/custom_cat_dog/cat1.jpg")


# img = cv.imread("./data_augmentation/data/cat1.jpg")

# transform = T.ToTensor()
# transform = T.Compose(
#     [
#         T.ToTensor(),
#         T.CenterCrop(250),
#         T.Pad(30, fill=0.5),
#     ]
# )


transform = T.Compose(
    [
        # T.ToPILImage(),
        T.RandAugment(),
        T.RandomHorizontalFlip(p=0.6),
        T.RandomVerticalFlip(p=0.3),
        T.ColorJitter(brightness=0.9),
        T.Resize((500, 500)),
        T.RandomRotation(degrees=30),
        T.ToTensor(),
        T.Normalize(mean=[0, 3, 0], std=[10, 1, 1]),  # important, also after ToTensor() must be apply
    ]
)

im_tensor = transform(img)
plt.imshow(im_tensor[0], cmap="gray")
plt.show()
