from os import environ
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
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
writer = SummaryWriter(log_dir="./mnist/runs/1")


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x_flatted = x.view(x.size(0), -1)
        z = self.fc1(x_flatted)
        a = self.relu(z)
        out = self.fc2(a)
        return out, x


model = NN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

sample_dataset_batch = next(iter(train_loader))
sample_input_batch = sample_dataset_batch[0]
sample_label_batch = sample_dataset_batch[1]

# writer.add_graph(model=model, input_to_model=sample_input_batch.view(-1,28*28))
writer.add_graph(model, sample_input_batch)

# def activation_hook(inst, inp, out):
#     # print("Here")
#     writer.add_histogram(repr(inst), out)


model.fc1.register_forward_hook(lambda module, input, output: writer.add_histogram(tag=repr(module), values=output))
model.fc2.register_forward_hook(lambda module, input, output: writer.add_histogram(tag="tag for hist", values=output))


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


# model.fc1.register_forward_hook(printnorm)

def train(model, criterion, optimizer):
    print("Training starts")
    step = 0
    for epoch in range(num_epochs):
        for batch_ndx, (x, y) in enumerate(train_loader):
            # x and y includes batch_size samples
            img_grid = torchvision.utils.make_grid(x)
            writer.add_image(tag=f"batch_iter: {batch_ndx}", img_tensor=img_grid)
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
                writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
                writer.add_scalar("loss", loss.item(), step)

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


train(model, criterion, optimizer)
accuracy(model, dataset=test_loader)
writer.close()
