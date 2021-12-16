from os import environ
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_prep import train_loader, test_loader

# Hyperparamethers
num_epochs = int(environ.get("NUM_EPOCHS"))
lr = float(environ.get("LEARNING_RATE"))


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out, x


model = CNN()
# print(model)


loss_func = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for i in range(num_epochs):
    for j, (x, y) in enumerate(train_loader):
        # x and y includes batch_size samples
        y_estimated_hat = model(x)[0]
        y_real = y
        loss = loss_func(model(x)[0], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {i+1}, loss after {j+1} batch {loss_func(y_estimated_hat,y_real)}")


print("Training was finished.")
current_time = time.strftime("%H-%M-%S")
path = f"./mnist/model/{current_time}.model"
torch.save(model.state_dict(), path)


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


def accuracy():
    f = 0
    total_samples = 0
    for i, (x, y) in enumerate(test_loader):  # if batch_size == total test samples, loop iterates one time.
        out = model(x)[0]
        total_samples = y.size()[0] + total_samples
        # print(total_samples)
        result = torch.argmax(input=out, dim=1)
        diff = torch.sub(y, result)
        f = torch.count_nonzero(diff) + f

    print(f"Total number of false estimation : {f}")
    print(f"Accuracy percent: {100- (100*f)/(total_samples)}")


accuracy()
