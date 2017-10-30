#!/usr/bin/env sh
set -e

TOOLS=$CAFFE_ROOT/build/tools

echo "Start Training..."

# Use pre-trained model on ImageNet
#$TOOLS/caffe train --solver=train.prototxt --weights=models/DenseNet_121.caffemodel

# Use pre-trained model on ImageNet
#$TOOLS/caffe train --solver=train.prototxt --weights=models/se_resnet_50_v1.caffemodel

# Train from scratch
$TOOLS/caffe train --solver=train.prototxt