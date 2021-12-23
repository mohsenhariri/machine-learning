from torch import nn
import torch
from data_prep import data_iter

# net = nn.Sequential(nn.Linear(2, 1))
net = nn.Sequential(nn.Linear(4, 2))


print("initial weights", net[0].weight.data)
print("initial bias", net[0].bias.data)

# we can also initial
# net[0].weight.data.normal_(0,0.01)
# net[0].bias.data.fill_(0)


loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 5

for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # l = loss(net(features),)
    print(f"epoch {epoch +1}, loss {l:f}")


print("final weights", net[0].weight.data)
print("final bias", net[0].bias.data)
