"""
Train data:  
Number of datapoints: 60000

Test Data:
Number of datapoints: 10000

Input:
1*28*28

Number of classes: 10
"""
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
    import torchvision
    import matplotlib.pyplot as plt

    print("train", train_data)
    print("test", test_data)
    print(train_loader.batch_size)

    sample_dataset_batch = next(iter(train_loader))
    sample_input_batch = sample_dataset_batch[0]
    sample_label_batch = sample_dataset_batch[1]

    print(sample_input_batch.size())
    print(sample_label_batch.size())

    img_grid = torchvision.utils.make_grid(sample_input_batch)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
