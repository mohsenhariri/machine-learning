"""
The argument values have priority over the environment variables.
If both of them are set, the environment variable is ignored.

python hyperparameters.py --batch-size 1 --batch-size-test 2 --epochs 3 --lr 4 --momentum 5 --reproducibility 6
source ./path/to/.env_variables && python hyperparameters.py
for example:
source ./mnist/.env_variables && python hyperparameters.py

for training models with arguments:
python main.py --batch-size 1 --batch-size-test 2 --epochs 3 --lr 4 --momentum 5 --reproducibility 6

for training models with environment variables:
source ./path/to/.env_variables && python main.py
"""
from os import environ
import argparse

try:
    batch_size = int(environ.get("BATCH_SIZE")) if environ.get("BATCH_SIZE") else 100
    batch_size_test = int(environ.get("BATCH_SIZE_TEST")) if environ.get("BATCH_SIZE_TEST") else 1000
    num_epochs = int(environ.get("NUM_EPOCHS")) if environ.get("NUM_EPOCHS") else 5
    lr = float(environ.get("LEARNING_RATE")) if environ.get("LEARNING_RATE") else 0.1
    momentum = float(environ.get("MOMENTUM")) if environ.get("MOMENTUM") else 0.9
    random_seed = int(environ.get("REPRODUCIBILITY")) if environ.get("REPRODUCIBILITY") else 777
except Exception as err:
    print("Error in parsing environment variables: ", repr(err))
    exit()


parser = argparse.ArgumentParser(description="Neural Network Models")

parser.add_argument("--batch-size", type=int, default=batch_size, metavar="N", help="input batch size for training")
parser.add_argument("--batch-size-test", type=int, default=batch_size_test, metavar="N", help="input batch size for testing")
parser.add_argument("--epochs", type=int, default=num_epochs, metavar="N", help="number of epochs to train")
parser.add_argument("--lr", type=float, default=lr, metavar="LR", help="learning rate")
parser.add_argument("--momentum", type=float, default=momentum, metavar="LR", help="momentum")
parser.add_argument("--reproducibility", type=int, default=random_seed, metavar="LR", help="reproducibility")

hp = parser.parse_args()


def main():
    print(hp.batch_size)
    print(hp.batch_size_test)
    print(hp.epochs)
    print(hp.lr)
    print(hp.momentum)
    print(hp.reproducibility)


if __name__ == "__main__":
    main()
