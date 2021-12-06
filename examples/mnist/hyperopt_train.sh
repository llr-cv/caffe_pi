#!/usr/bin/env sh
set -e

python hyperopt_train.py --solver lenet_solver.prototxt --gpus 1 2 3 2>&1 | tee hyperopt_train.log &
