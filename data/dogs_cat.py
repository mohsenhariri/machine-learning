from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy
import random


train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


train_data_path = r"./data/dogs-vs-cats/train/"

test_data_path = r"./data/dogs-vs-cats/test1/"

train_image_paths = []  # to store image paths in list
classes = []  # to store class values

# for data_path in glob.glob(train_data_path + '/*'):
#    label = data_path.split('/')[-1]
   
#     classes.append(data_path.split('/')[-1])
#     train_image_paths.append(glob.glob(data_path + '/*'))
    
for i in glob.glob(train_data_path+'*.jpg'):
   print(i)

print(classes)

# train_image_paths = list(flatten(train_image_paths))
# random.shuffle(train_image_paths)

# print('train_image_path example: ', train_image_paths[0])
# print('class example: ', classes[0])