import os.path as path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

__all__ = ["train_loader", "test_loader"]


class BuildDataset(Dataset):
    def __init__(self, csv_file, root, transform=None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform
        pass

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # return super().__getitem__(index)
        img_path = path.join(self.root, self.annotations.iloc[index, 0])
        # print(img_path)
        img = Image.open(img_path)
        label = torch.tensor(self.annotations.iloc[index, 1])
        if self.transform:
            img = self.transform(img)

        return (img, label)

    def __repr__(self) -> str:
        return "dataset"


cats_dogs_dataset = BuildDataset(csv_file="./data/custom_cat_dog/resize/label.csv", root="./data/custom_cat_dog/resize", transform=T.ToTensor())


train_loader = DataLoader(dataset=cats_dogs_dataset, shuffle=True, batch_size=2)
test_loader = DataLoader(dataset=cats_dogs_dataset, shuffle=True, batch_size=3)

# for batch_ndx, (x, y) in enumerate(train_loader):
#     print(x.size())
# x and y includes batch_size samples
# img_grid = torchvision.utils.make_grid(x)
# writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
# forward
# y_hat = model(x)
# loss = criterion(y_hat, y)
# backwards


# exit()


# transfor = T.Compose([T.ToPILImage(), T.RandAugment()])

# dataset_augmented = BuildDataset(csv_file="./data/custom_cat_dog/data/label.csv", root="./data/custom_cat_dog/data", transform=T.ToTensor())

# for x, y in cats_dogs_dataset:
#     print(x.size())
#     print(y)
    # exit()
