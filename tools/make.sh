#!/usr/bin/env bash
set -e

if [ ! -n "$1" ]; then
    ARCH="sm_35"
else
    ARCH=$1
fi
CURDIR=$(cd "$(dirname "$0")";pwd)
echo ${CURDIR}

# make utils/nms
cd ${CURDIR}/../lib/utils

python setup.py  build_ext --inplace
rm -rf build

# make layers/roi_pooling
cd ${CURDIR}/../lib/layers/roi_pooling/src/cuda

echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=${ARCH}

cd ../../
python build.py

# make layers/loss/warp_smooth_l1_loss
cd ${CURDIR}/../lib/layers/loss/warp_smooth_l1_loss/src/cuda

echo "Compiling warp smooth l1 loss kernels by nvcc..."
nvcc -c -o warp_smooth_l1_loss_kernel.cu.o warp_smooth_l1_loss_kernel.cu -x cu -Xcompiler -fPIC -arch=${ARCH}

cd ../../
python build.py
