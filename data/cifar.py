from os import path
from torchvision import datasets, transforms
from torch.utils import data
from hyperparameters import hp

__all__ = ["train_loader", "test_loader"]


transform = transforms.ToTensor()

# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ]
# )

file_exists = path.exists("./data/CIFAR10")

train_data = datasets.CIFAR10(root="./data/CIFAR10", train=True, transform=transform, download=not file_exists)
# print(train_data)
### Number of datapoints: 50000

test_data = datasets.CIFAR10(root="./data/CIFAR10", transform=transform, train=False)
# print(test_data)
### Number of datapoints: 10000

train_loader = data.DataLoader(dataset=train_data, batch_size=hp.batch_size, shuffle=True)
# print(train_loader.batch_size)
test_loader = data.DataLoader(dataset=test_data, batch_size=10000)  # default batch_size is 1


def main():
    import torchvision
    import matplotlib.pyplot as plt

    sample_dataset_batch = next(iter(train_loader))
    sample_input_batch = sample_dataset_batch[0]
    sample_label_batch = sample_dataset_batch[1]

    print(sample_input_batch.size())
    print(sample_label_batch.size())

    img_grid = torchvision.utils.make_grid(sample_input_batch)
    print(img_grid.size())
    print(img_grid.permute(1, 2, 0).size())
    plt.imshow(img_grid.permute(1, 2, 0))
    """
    about permute: 
    https://stackoverflow.com/questions/51329159/how-can-i-generate-and-display-a-grid-of-images-in-pytorch-with-plt-imshow-and-t
    torchvision.utils.make_grid() returns a tensor which contains the grid of images. But the channel dimension has to be moved to the end since that's what matplotlib recognizes
    """
    plt.show()


if __name__ == "__main__":
    main()
