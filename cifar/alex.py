from os import environ, path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from architectures.AlexNet.net import AlexNet
from data.cifar import train_loader, test_loader

# Hyperparamethers
try:
    num_epochs = int(environ.get("NUM_EPOCHS"))
    lr = float(environ.get("LEARNING_RATE"))
    random_seed = int(environ.get("REPRODUCIBILITY"))
    momentum = int(environ.get("MOMENTUM"))
except:
    num_epochs = 1
    lr = 0.1
    random_seed = 777
    momentum = 0.8
torch.manual_seed(random_seed)

j = 0
while True:
    j += 1
    file_exists = path.exists(f"./cifar/runs/{j}")
    if not file_exists:
        break

writer = SummaryWriter(log_dir=f"./cifar/runs/{j}")


# model = AlexNet(num_classes=10,input_size=(32,32))
model = AlexNet(num_classes=10,input_size=(224,224))



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), momentum=momentum, lr=lr)


# exit()
def train(model, criterion, optimizer, data_loader):
    print("Training starts")
    step = 0
    for epoch in range(num_epochs):
        for batch_ndx, (x, y) in enumerate(data_loader):
            # forward
            y_hat = model(x)
            loss = criterion(y_hat, y)
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            print(f"Epoch: {epoch+1}, loss after {batch_ndx+1} bach {loss}")
            if (batch_ndx + 1) % 20 == 0:
                step = step + batch_ndx

        print(f"Epoch {epoch+1} was finished.")

    print("Training was finished.")


def accuracy(model, dataset):
    f = 0
    total_samples = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            out = model(x)
            total_samples += y.size()
            result = torch.argmax(input=out, dim=1)
            diff = torch.sub(y, result)
            f += torch.count_nonzero(diff)
    acc = 100 - (100 * f) / (total_samples)
    print(f"Total number of false estimation : {f}")
    print(f"Accuracy percent: {acc}")
    return acc


train(model, criterion, optimizer, train_loader)
accuracy(model, test_loader)
