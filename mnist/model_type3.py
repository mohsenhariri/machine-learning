from os import environ
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prep import train_loader, test_loader

random_seed = int(environ.get("REPRODUCIBILITY"))
torch.manual_seed(random_seed)

# Hyperparamethers
num_epochs = int(environ.get("NUM_EPOCHS"))
lr = float(environ.get("LEARNING_RATE"))
momentum = 0.5


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
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out, x


model = CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
