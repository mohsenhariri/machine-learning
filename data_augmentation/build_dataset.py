import os.path as path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


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


cats_dogs_dataset = BuildDataset(
    csv_file="./data_augmentation/data/label.csv", root="./data_augmentation/data", transform=T.ToTensor()
)

# print(cats_dogs_dataset)

train_loader = DataLoader(dataset=cats_dogs_dataset, shuffle=True, batch_size=2)
test_loader = DataLoader(dataset=cats_dogs_dataset, shuffle=True, batch_size=1)

# exit()


def main():
    pass


if __name__ == "__main__":
    main()


transfor = T.Compose([T.ToPILImage(), T.RandAugment()])

dataset_augmented = BuildDataset(
    csv_file="./data_augmentation/data/label.csv", root="./data_augmentation/data", transform=T.ToTensor()
)

for x, y in cats_dogs_dataset:
    print(x.size())
    print(y)
    # exit()
