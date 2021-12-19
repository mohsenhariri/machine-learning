from os import environ
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prep import train_loader, test_loader


# Hyperparamethers
num_epochs = int(environ.get("NUM_EPOCHS"))
lr = float(environ.get("LEARNING_RATE"))
momentum = float(environ.get("MOMENTUM"))

random_seed = int(environ.get("REPRODUCIBILITY"))
torch.manual_seed(random_seed)


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        y_hat = F.log_softmax(x, dim=0)
        return y_hat, x


model = CNN()
# print(model)

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
