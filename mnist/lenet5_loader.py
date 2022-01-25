import torch
from architectures.LeNet5.net import LeNet5
from data.mnist import train_loader, test_loader


model = LeNet5()

model.load_state_dict(torch.load("./mnist/saved_models/17-54-17_dict.pth"))


def accuracy(model, dataset) -> torch.Tensor:
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


accuracy(model, test_loader)
