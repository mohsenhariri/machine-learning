from os import environ, path
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from data_prep import train_loader, test_loader

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
    file_exists = path.exists(f"./cifar/runs/{j}")
    if not file_exists:
        break

writer = SummaryWriter(log_dir=f"./cifar/runs/{j}")


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=6 * 16 * 16, out_features=10)  # 1536
        # self.fc1 = nn.Linear(in_features=1536, out_features=10)
        # self.fc1 = nn.Linear(in_features=3750, out_features=10)

        # 3750
        # paramethers 3*2*2*6 + 6 
    def forward(self, x):
        out_conv1 = self.conv1(x)
        # x = x.view(x.size()[0], -1)
        x = torch.flatten(out_conv1, 1)  # flatten all dimensions except batch

        x = self.fc1(x)
        return x, out_conv1


model = ConvNet()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

# def initialize_weights(m):
#   if isinstance(m, nn.Conv2d):
#       nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
#       if m.bias is not None:
#           nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.BatchNorm2d):
#       nn.init.constant_(m.weight.data, 1)
#       nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.Linear):
#       nn.init.kaiming_uniform_(m.weight.data)
#       nn.init.constant_(m.bias.data, 0)


model.apply(init_weights)

# nn.init.xavier_normal_(model.fc1.weight)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

# batch size is 100
# summary(model=model, input_size=(train_loader.batch_size, 3, 32, 32))
# print(train_loader.batch_size)
summary(model=model, input_size=(100, 3, 32, 32))

exit()
# print(model)
# print(model.conv1.parameters().data)

# print(model.conv1.weight.size())
# print(model.conv1.bias.size())

# for batch_ndx, (x, y) in enumerate(test_loader):
#     y_hat = model(x)[1]
#     print(y_hat)
#     exit()


# exit()
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
            print(f"Epoch: {epoch+1}, loss after {batch_ndx+1} bach {loss}")
            if (batch_ndx + 1) % 20 == 0:
                step = step + batch_ndx
                # writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
                img_grid = torchvision.utils.make_grid(x)
                writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)

                conv_output = model(x)[1]
                for channel_idx in range(6):

                    img_grid_conv = torchvision.utils.make_grid(conv_output[:, channel_idx, :, :].reshape([100, 1, 16, 16]))
                    # print(conv_output[:, channel_idx, :, :].reshape([100,1,16,16]).size())
                    # exit()

                    writer.add_image(tag=f"batch_iter: {batch_ndx} Channel: {channel_idx}", img_tensor=img_grid_conv)

                # exit()
                # img_grid_conv = torchvision.utils.make_grid(model(x)[1])
                # writer.add_image(tag=f"batch_iter: {batch_ndx}_CONV", img_tensor=img_grid_conv)

                writer.add_scalar("loss", loss.item(), step)

        print(f"Epoch {epoch+1} was finished.")

    print("Training was finished.")


def accuracy(model, dataset):
    f = 0
    total_samples = 0
    with torch.no_grad():
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


train(model, criterion, optimizer, train_loader)
# accuracy(model, test_loader)
