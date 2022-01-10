#!/bin/bash

rm ./mnist/models/*.pth # delete all saved models
rm -rf ./mnist/runs/* # delete all saved tensorboard data

rm -rf ./cifar/runs/*

export PYTHONPATH="${PYTHONPATH}:."

# source ./mnist/.env_variables && python ./mnist/model_type1.py


### Python Setup

python setup.py sdist

tensorboard --logdir=./mnist/runs
