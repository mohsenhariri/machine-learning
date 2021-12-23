import torch
import torch.nn as nn


torch.manual_seed(555)

lr = 0.1


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=10)

    def forwad(self, x):
        out = self.fc1(x)
        return out


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)


epoch = 3
for j in range(epoch):
    for i, (x, y) in enumerate(dataset):
        # forwad
        y_hat = model(x)
        loss = criterion(y_hat - y)

        # backward
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
        print(f"Epoch: {i+1}, loss after {j+1} bach {loss}")
