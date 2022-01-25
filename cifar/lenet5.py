from os import path
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from architectures.LeNet5.net import LeNet5
from data.cifar_resized import train_loader, test_loader
from hyperparameters import hp


torch.manual_seed(hp.reproducibility)

j = 0
while True:
    j += 1
    file_exists = path.exists(f"./cifar/runs/lenet5-{j}")
    if not file_exists:
        break

writer = SummaryWriter(log_dir=f"./cifar/runs/lenet5-{j}")

model = LeNet5(num_classes=10, is_colored=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=hp.lr)


def train(model, criterion, optimizer, log=False):
    print("Training starts.")
    step = 0
    for epoch in range(hp.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f"Epoch {epoch+1}/{hp.epochs}")

        for batch_idx, (x, y) in loop:  # enumerate(tran_loader)
            ## forward
            y_hat = model(x)
            loss = criterion(y_hat, y)
            ## backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## log
            if log:
                # img_grid = torchvision.utils.make_grid(x)
                # writer.add_image(tag=f"batch_iter: {batch_idx}", img_tensor=img_grid)
                # print(f"Epoch: {epoch+1}, loss after {batch_idx+1} bach {loss}")
                if (batch_idx + 1) % 20 == 0:
                    step = step + batch_idx
                    writer.add_scalar("accuracy", accuracy(model, dataset=test_loader).item(), step)
                    writer.add_scalar("loss", loss.item(), step)

        print(f"Accuracy after epoch {epoch+1}: {accuracy(model, dataset=test_loader).item()}")

    print("Training was finished.")


def accuracy(model, dataset) -> torch.Tensor:
    f = 0
    total_samples = 0
    for i, (x, y) in enumerate(dataset):
        out = model(x)
        total_samples += y.size()[0]
        result = torch.argmax(input=out, dim=1)
        diff = torch.sub(y, result)
        f += torch.count_nonzero(diff)
    acc = 100 - (100 * f) / (total_samples)
    print(f"Total number of false estimation : {f}")
    # print(f"Accuracy percent: {acc}")
    return acc


def save_model(model):
    import time

    current_time = time.strftime("%H-%M-%S")
    torch.save(model.state_dict(), f"./cifar/saved_models/{current_time}_dict.pth")
    print("Model was saved.")


train(model, criterion, optimizer, log=False)
accuracy(model, test_loader)
save_model(model)
writer.close()
