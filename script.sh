#!/bin/bash

rm ./mnist/models/*.pth # delete all saved models
rm -rf ./mnist/runs/* # delete all saved tensorboard data

rm -rf ./cifar/runs/*