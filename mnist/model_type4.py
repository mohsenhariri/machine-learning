from collections import OrderedDict
from os import environ
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prep import train_loader, test_loader


# Hyperparameters
num_epochs = int(environ.get("NUM_EPOCHS"))
lr = float(environ.get("LEARNING_RATE"))
momentum = float(environ.get("MOMENTUM"))

random_seed = int(environ.get("REPRODUCIBILITY"))
torch.manual_seed(random_seed)

# model = nn.Sequential()


layer1conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
out1relu = nn.ReLU()
layer2pool = nn.MaxPool2d(kernel_size=2)
layer3conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
out3relu = nn.ReLU()
layer4pool = nn.MaxPool2d(kernel_size=2)
layer5fc = nn.Linear(32 * 5 * 5, 10)

model = nn.Sequential(
    OrderedDict(
        [
            ("layer1conv", layer1conv),
            ("o ut1relu", out1relu),
            ("layer2pool", layer2pool),
            ("layer3conv", layer3conv),
            ("out3relu", out3relu),
            ("layer4pool", layer4pool),
            ("layer5fc", layer5fc),
        ]
    )
)


print(model)



loss_func = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


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
