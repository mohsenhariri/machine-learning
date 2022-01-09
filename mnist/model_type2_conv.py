from os import environ
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from data_prep import train_loader, test_loader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
writer = SummaryWriter(log_dir="./mnist/runs/1")


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=10, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=10 * 14 * 14, out_features=10)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=100, out_features=10)
        # self.fc3 = nn.Linear(in_features=50, out_features=25)
        # self.fc4 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        # img_grid = torchvision.utils.make_grid(x)
        # writer.add_image(tag="orig", img_tensor=img_grid)

        out = self.conv1(x)
        # print("size on input after conv1", out.size())
        # for channel_number in range(10):
        #     img_grid = torchvision.utils.make_grid(out[:, channel_number, :, :].reshape(100, 1, 14, 14))
        #     writer.add_image(tag=f"channel: {channel_number}", img_tensor=img_grid)

        # print(out.size())

        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        # exit()
        # writer.close()

        # exit()

        return out, x


model = NN()


print(model.conv1.weight.size())
print(model.conv1.bias.size())
# print(model.fc2.bias.size())

# print(model.fc1.weight.size())
# print(model.fc1.bias.size())

exit()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

# sample_dataset_batch = next(iter(train_loader))
# sample_input_batch = sample_dataset_batch[0]
# sample_label_batch = sample_dataset_batch[1]

# writer.add_graph(model=model, input_to_model=sample_input_batch.view(-1,28*28))
# writer.add_graph(model, sample_input_batch)


# def activation_hook(inst, inp, out):
#     # print("Here")
#     writer.add_histogram(repr(inst), out)


# model.conv1.register_forward_hook(lambda inst, inp, out: writer.add_histogram(repr(inst), out))

# model.conv1.register_forward_hook(lambda inst, inp, out: writer.add_histogram(repr(inst), out))


# def activation_hook(module, input, output):
#     for channel_number in range(10):
#         img_grid = torchvision.utils.make_grid(output[:, channel_number, :, :].reshape(100, 1, 14, 14))
#         writer.add_image(tag=f"channel: {channel_number}", img_tensor=img_grid)
    # print("ok")
    # for channel_number in range(10):
    #     with torch.no_grad():
    #         x = output[:, channel_number, :, :]
    #         # print('XX',x.size())
    #         # exit()
    #         # x2 = x.view(100, 1, 14, 14)
    #         xxx = torch.reshape(x, (100, 1, 14, 14))
    #         img_grid = torchvision.utils.make_grid(xxx)
    #         writer.add_image(tag=f"channel: {channel_number} iter{3}", img_tensor=img_grid)


# with torch.no_grad():
# Img_.resize_(Img.size()).copy_(Img))
# model.conv1.register_forward_hook(activation_hook)


def train(model, criterion, optimizer):
    print("Training starts")
    step = 0
    for epoch in range(num_epochs):
        for batch_ndx, (x, y) in enumerate(train_loader):
            # x and y includes batch_size samples
            # print(x.size())
            # img_grid = torchvision.utils.make_grid(x)
            # writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
            # forward
            y_hat = model(x)[0]
            loss = criterion(y_hat, y)
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            print(f"Epoch: {epoch+1}, loss after {batch_ndx+1} bach {loss}")

            def activation_hook(module, input, output):
                for channel_number in range(1):
                    img_grid = torchvision.utils.make_grid(output[:, channel_number, :, :].reshape(100, 1, 14, 14))
                    writer.add_image(tag=f"Batch: {batch_ndx},channel: {channel_number}", img_tensor=img_grid)

            # model.conv1.register_forward_hook(activation_hook)

            if (batch_ndx + 1) % 20 == 0:
                print("logging")
                step = step + batch_ndx
                # writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
                writer.add_scalar("loss", loss.item(), step)
                model.conv1.register_forward_hook(activation_hook)


        print(f"Epoch {epoch+1} was finished.")

    print("Training was finished.")


def accuracy(model, dataset):
    f = 0
    total_samples = 0
    with torch.no_grad():
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


train(model, criterion, optimizer)
accuracy(model, dataset=test_loader)
writer.close()
