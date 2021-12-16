from os import path, environ
from torchvision import datasets, transforms
from torch.utils import data

batch_size = int(environ.get("BATCH_SIZE"))

transform = transforms.ToTensor()
file_exists = path.exists("./data/MNIST")

train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=not file_exists)
# print(train_data)
### Number of datapoints: 60000

test_data = datasets.MNIST(root="./data", transform=transform, train=False)
### Number of datapoints: 10000

train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=10000)  # default batch_size is 1
