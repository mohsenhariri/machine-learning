from build_dataset import train_loader
import matplotlib.pyplot as plt

print(train_loader)

for x, y in train_loader:
    # print(x[0][0])
    # print(y.item())
    plt.title(y.item())
    plt.imshow(x[0][0], cmap="gray")
    plt.show()

    # exit()
