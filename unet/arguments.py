import argparse

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
hp = parser.parse_args() # Hyperparamethers


def main():
    print(hp)


if __name__ == "__main__":
    main()
