from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T

# from torchvision.transforms.transforms import ToTensor


def show2(im1, im2):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(im1, cmap="gray")
    ax[1].imshow(im2, cmap="gray")
    plt.show()


def show3(im1, im2, im3):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(im1, cmap="gray")
    ax[1].imshow(im2, cmap="gray")
    ax[2].imshow(im3, cmap="gray")
    plt.show()


def show4(im1, im2, im3, im4):
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(im1, cmap="gray")
    ax[0][1].imshow(im2, cmap="gray")
    ax[1][0].imshow(im3, cmap="gray")
    ax[1][1].imshow(im4, cmap="gray")
    plt.show()


img = Image.open("./data_augmentation/data/cat1.jpg")


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
exit()
# print(im_tensor.size())
maxPooing_function = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))


im_pooled = maxPooing_function(im_tensor)

show2(im_tensor[0], im_pooled[0])
