#!/usr/bin/env sh
set -e

../../build/tools/caffe train --solver=../../examples/mnist/lenet_solver.prototxt --gpu 1 2>&1 | tee lenet_mnist.log &
