from os import environ, path
from architectures.AlexNet.net import AlexNet
from .data_prep import train_loader, test_loader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Hyperparamethers
try:
    num_epochs = int(environ.get("NUM_EPOCHS"))
    lr = float(environ.get("LEARNING_RATE"))
    random_seed = int(environ.get("REPRODUCIBILITY"))
except:
    num_epochs = 1
    lr = 0.1
    random_seed = 777

torch.manual_seed(random_seed)

j = 0
while True:
    j += 1
    file_exists = path.exists(f"./mnist/runs/{j}")
    if not file_exists:
        break

writer = SummaryWriter(log_dir=f"./mnist/runs/{j}")


model = AlexNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)


def train(model, criterion, optimizer):
    print("Training starts.")
    step = 0
    for epoch in range(num_epochs):
        for batch_ndx, (x, y) in enumerate(train_loader):
            # x and y includes batch_size samples
            ## forward
            y_hat = model(x)
            exit()
            loss = criterion(y_hat, y)
            ## backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## log
            # img_grid = torchvision.utils.make_grid(x)
            # writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
            print(f"Epoch: {epoch+1}, loss after {batch_ndx+1} bach {loss}")
            if (batch_ndx + 1) % 20 == 0:
                step = step + batch_ndx
                writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
            #     writer.add_scalar("loss", loss.item(), step)

        print(f"Epoch {epoch+1} was finished.")

    print("Training was finished.")


def accuracy(model, dataset):
    f = 0
    total_samples = 0
    for i, (x, y) in enumerate(dataset):  # if batch_size == total test samples, loop iterates one time.
        out = model(x)
        total_samples += y.size()[0]
        # print(y.size()[0]) this is batch size
        result = torch.argmax(input=out, dim=1)
        diff = torch.sub(y, result)
        f += torch.count_nonzero(diff)
    acc = 100 - (100 * f) / (total_samples)
    print(f"Total number of false estimation : {f}")
    print(f"Accuracy percent: {acc}")
    return acc


train(model, criterion, optimizer)
accuracy(model, test_loader)
