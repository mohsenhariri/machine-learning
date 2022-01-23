from os import path
from torchvision import datasets, transforms
from torch.utils import data
from hyperparameters import hp


__all__ = ["train_loader", "test_loader"]


transform = transforms.ToTensor()
file_exists = path.exists("./data/MNIST")

train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=not file_exists)
### Number of datapoints: 60000

test_data = datasets.MNIST(root="./data", transform=transform, train=False)
### Number of datapoints: 10000

train_loader = data.DataLoader(dataset=train_data, batch_size=hp.batch_size, shuffle=True)

test_loader = data.DataLoader(dataset=test_data, batch_size=10000)  # default batch_size is 1


def main():
    print("train", train_data)
    print("test", test_data)
    print(train_loader.batch_size)


if __name__ == "__main__":
    main()
