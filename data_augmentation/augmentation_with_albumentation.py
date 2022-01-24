# https://albumentations.ai/docs/#introduction-to-image-augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./data/custom_cat_dog/cat1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# transformer = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=350),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
#         A.RandomCrop(height=256, width=256),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.2),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#         ToTensorV2(),
#     ]
# )

transformer = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ]
)


im_trans = transformer(image=img)

im_tensor = transformed_image = im_trans["image"]

plt.imshow(im_tensor.permute(1, 2, 0))
plt.show()
