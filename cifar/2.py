from os import environ, path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from data_prep import train_loader, test_loader

# Hyperparamethers
try:
    num_epochs = int(environ.get("NUM_EPOCHS"))
    lr = float(environ.get("LEARNING_RATE"))
    momentum = float(environ.get("MOMENTUM"))
    random_seed = int(environ.get("REPRODUCIBILITY"))
except:
    num_epochs = 1
    lr = 0.1
    random_seed = 777

torch.manual_seed(random_seed)

j = 0
while True:
    j += 1
    file_exists = path.exists(f"./cifar/runs/{j}")
    if not file_exists:
        break

writer = SummaryWriter(log_dir=f"./cifar/runs/{j}")


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        # Layer definitions
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv1N = F.relu(out_conv1)
        out_pool1 = self.pool(out_conv1N)

        out_conv2 = self.conv2(out_pool1)
        out_conv2N = F.relu(out_conv2)
        out_pool2 = self.pool(out_conv2N)

        out_flatted = torch.flatten(out_pool2, 1)  # flatten all dimensions except batch
        z1 = self.fc1(out_flatted)
        a1 = F.relu(z1)
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        out = self.fc3(a2)
        return out, (out_conv1N, out_conv2N)


model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

# batch size is 100
# summary(model=model, input_size=(train_loader.batch_size, 3, 32, 32))

# print(model)
# print(model.conv1.parameters())
# print(model.conv1.weight.size())
# print(model.conv1.bias.size())


def train(model, criterion, optimizer, data_loader):
    print("Training starts")
    step = 0
    for epoch in range(num_epochs):
        for batch_ndx, (x, y) in enumerate(data_loader):
            # x and y includes batch_size samples
            # img_grid = torchvision.utils.make_grid(x)
            # writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
            # print(f"{batch_ndx} saved.")

            # forward
            y_hat = model(x)[0]
            loss = criterion(y_hat, y)
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            # print(f"Epoch: {epoch+1}, loss after {batch_ndx+1} bach {loss}")
            if (batch_ndx + 1) % 20 == 0:
                step = step + batch_ndx
                writer.add_scalar("loss", loss.item(), step)
                # writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
                # img_grid = torchvision.utils.make_grid(x)
                # writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)

                # conv_output = model(x)[1]
                # for channel_idx in range(6):

                # img_grid_conv = torchvision.utils.make_grid(conv_output[:, channel_idx, :, :].reshape([100, 1, 16, 16]))
                # print(conv_output[:, channel_idx, :, :].reshape([100,1,16,16]).size())
                # exit()

                # writer.add_image(tag=f"batch_iter: {batch_ndx} Channel: {channel_idx}", img_tensor=img_grid_conv)

                # exit()
                # img_grid_conv = torchvision.utils.make_grid(model(x)[1])
                # writer.add_image(tag=f"batch_iter: {batch_ndx}_CONV", img_tensor=img_grid_conv)

        print(f"Epoch {epoch+1} was finished.")

    print("Training was finished.")


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
    acc = 100 - (100 * f) / (total_samples)
    print(f"Total number of false estimation : {f}")
    print(f"Accuracy percent: {acc}")
    return acc


train(model, criterion, optimizer, train_loader)
accuracy(model, test_loader)
