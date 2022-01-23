from os import environ
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data_prep import train_loader, test_loader

# Hyperparameters
num_epochs = int(environ.get("NUM_EPOCHS"))
lr = float(environ.get("LEARNING_RATE"))
random_seed = int(environ.get("REPRODUCIBILITY"))

torch.manual_seed(random_seed)
writer = SummaryWriter(log_dir="./mnist/runs")


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out, x


class CNN2(nn.Module):
    def __init__(self) -> None:
        super(CNN2, self).__init__()
        self.NonLinearlayers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.Linearlayers = nn.Sequential(
            nn.Linear(
                in_features=32 * 7 * 7,
                out_features=10,
            ),
        )

    def forward(self, x):
        x = self.NonLinearlayers(x)
        x = x.view(x.size(0), -1)
        out = self.Linearlayers(x)
        return out, x


class CNN3(nn.Module):
    def __init__(self) -> None:
        super(CNN3, self).__init__()
        self.Layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 7 * 7,
                out_features=10,
            ),
        )

    def forward(self, x):
        out = self.Layers(x)
        return out, x


class CNN4(nn.Module):
    def __init__(self) -> None:
        super(CNN4, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1))
        # self.layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.layer(x)
        return out, x


model = CNN3()
# print("Model structure: ", model, "\n\n")


criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# for i,params in enumerate(model.parameters()):
# for i, (name, param) in enumerate(model.named_parameters()):
#     # print(param.data)
#     print(
#         f"""
#     \n{i} Iteration,
#     \n Layer: {name}
#     \n Size: {param.size()}
#     \n Values : {param.data}"""
#     # \n Values : {param[:2]}"""

#     )


def train(model, criterion, optimizer):
    print("Training starts")
    for i in range(num_epochs):
        for j, (x, y) in enumerate(train_loader):
            # x and y includes batch_size samples
            # forward
            y_hat = model(x)[0]
            loss = criterion(y_hat, y)
            # log
            writer.add_scalar("Loss/train", loss, j)
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Epoch: {i+1}, loss after {j+1} bach {loss}")
        print(f"Epoch {i+1} was finished.")

    print("Training was finished.")


def save_model():
    current_time = time.strftime("%H-%M-%S")
    torch.save(model.state_dict(), f"./mnist/models/{current_time}_dict.pth")
    torch.save(model, f"./mnist/models/{current_time}.pth")
    print("Model was saved.")


def load_model(path):
    model = torch.load(f"./mnist/models/{path}")
    # model.eval()

    # print("Architecture of network is: ", model)
    return model
    # print(model.parameters())
    # for params in model.parameters():
    #     print(params)

    # for params in model.biases():
    #     print(params)

    # for params in model.parameters():
    #     print(params)


def show(img, y_estimate, y_label):
    plt.imshow(img, cmap="gray")
    plt.title(f"Estimate: {y_estimate}, Label: {y_label}")
    plt.show()


def accuracy_v1():  ## if batch size is 1.
    f = 0
    for i, (x, y) in enumerate(test_loader):
        model_output = model(x)[0]
        y_hat = torch.argmax(model_output)
        if y_hat.item() != y.item():
            show(x[0][0], y_hat.item(), y.item())
            f = f + 1

    print(f"Total number of false estimation : {f}")
    print(f"Percent: {100- (100*f)/(i+1)}")


def accuracy(model, dataset):
    f = 0
    total_samples = 0
    for i, (x, y) in enumerate(dataset):  # if batch_size == total test samples, loop iterates one time.
        out = model(x)[0]
        total_samples += y.size()[0]
        # print(y.size()[0]) this is batch size
        result = torch.argmax(input=out, dim=1)
        diff = torch.sub(y, result)
        f += torch.count_nonzero(diff)

    print(f"Total number of false estimation : {f}")
    print(f"Accuracy percent: {100- (100*f)/(total_samples)}")


# exit()

train(model, criterion, optimizer)
# save_model()
# model = load_model("22-50-53.pth")
accuracy(model, test_loader)

# print()
writer.flush()
writer.close()
