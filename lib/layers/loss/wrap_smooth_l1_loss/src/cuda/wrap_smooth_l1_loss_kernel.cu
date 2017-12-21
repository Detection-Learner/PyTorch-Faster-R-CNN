/*
 *  PyTorch Faster R-CNN
 *  Written by Yuanshun Cui
*/
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "wrap_smooth_l1_loss_kernel.h"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void WrapSmoothL1LossForward(const int nthreads, const float sigma2, const int size_average,
                               const float * data_inputs, const float * data_targets,
                               const float * data_inside_weights, const float * data_outside_weights,
                               float * output_flat)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        float val = data_inside_weights[index] * (data_inputs[index] - data_targets[index]);
        float abs_val = abs(val);
        if (abs_val < 1.0 / sigma2)
        {
            output_flat[index] = data_outside_weights[index] * 0.5 * val * val * sigma2;
        }
        else
        {
            output_flat[index] = data_outside_weights[index] * (abs_val - 0.5 / sigma2);
        }

    }
}

int WrapSmoothL1LossForwardLaucher(const int data_num, const float sigma2, const int size_average,
                               const float * data_inputs, const float * data_targets,
                               const float * data_inside_weights, const float * data_outside_weights,
                               float * output_flat, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    cudaError_t err;

    WrapSmoothL1LossForward<<<(data_num + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        data_num, sigma2, size_average, data_inputs, data_targets, data_inside_weights, data_outside_weights,
        output_flat);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

__global__ void WrapSmoothL1LossBackward(const int nthreads, const float sigma2, const int size_average,
                               const float * data_inputs, const float * data_targets,
                               const float * data_inside_weights, const float * data_outside_weights,
                               float * input_flat1, float * input_flat2,
                               const float * output_flat)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        float val = data_inside_weights[index] * (data_inputs[index] - data_targets[index]);
        float abs_val = abs(val);
        if (abs_val < 1.0 / sigma2)
        {
            input_flat1[index] = data_outside_weights[index] * data_inside_weights[index] * val * sigma2;
        }
        else
        {
            input_flat1[index] = data_outside_weights[index] * data_inside_weights[index] * ((0<val) - (val<0));
        }
        input_flat2[index] = -input_flat1[index] * output_flat[0] / (float)size_average;
        input_flat1[index] = input_flat1[index] * output_flat[0] / (float)size_average;
    }
}

int WrapSmoothL1LossBackwardLaucher(const int data_num, const float sigma2, const int size_average,
                               const float * data_inputs, const float * data_targets,
                               const float * data_inside_weights, const float * data_outside_weights,
                               float * input_flat1, float * input_flat2,
                               const float * output_flat, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    cudaError_t err;

    WrapSmoothL1LossBackward<<<(data_num + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        data_num, sigma2, size_average, data_inputs, data_targets, data_inside_weights, data_outside_weights,
        input_flat1, input_flat2, output_flat);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

#ifdef __cplusplus
}
#endif
