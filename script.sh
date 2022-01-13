#!/bin/bash

rm ./mnist/models/*.pth # delete all saved models
rm -rf ./mnist/runs/* # delete all saved tensorboard data

rm -rf ./cifar/runs/*

export PYTHONPATH="${PYTHONPATH}:."



### Python Setup

python setup.py sdist

source ./mnist/.env_variables && python main.py
source ./cifar/.env_variables && python main.py


tensorboard --logdir=./mnist/runs
