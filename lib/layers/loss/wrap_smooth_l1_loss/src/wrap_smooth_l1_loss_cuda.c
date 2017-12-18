/*
 * PyTorch Faster R-CNN
 * Written by Hongyu Pan, Yuanshun Cui
 *
 */

#include <THC/THC.h>
#include <math.h>
#include "cuda/wrap_smooth_l1_loss_kernel.h"
#include <stdio.h>

extern THCState *state;

int wrap_smooth_l1_loss_forward_cuda(float sigma, int size_average,
                    THCudaTensor * inputs, THCudaTensor * targets,
                    THCudaTensor * inside_weights, THCudaTensor * outside_weights, THCudaTensor * output)
{
    // check size
    if (!THCudaTensor_isSameSizeAs(state, inputs, targets))
        return 0;
    THCudaTensor * unsumLoss = THCudaTensor_new(state);//THCudaTensor_newWithTensor(state, inputs);
		THCAssertSameGPU(THCudaTensor_checkGPU(state, 2, unsumLoss, inputs));
		THCudaTensor_resizeAs(state, unsumLoss, inputs);
		THCudaTensor_zero(unsumLoss, inputs);
		/*
		 *
		 *  < 0.3 not support THCudaTensor_zerosLike() 
		 *  
		*/
    //THCudaTensor_zerosLike(state, unsumLoss, inputs);
    //THCudaTensor_fill(state, unsumLoss, 0);
    // data dim
    int data_dim = inputs->nDimension;
    int d=0, data_num=1;
    for (d=0; d < data_dim; d++)
    {
        data_num *= THCudaTensor_size(state, inputs, d);
    }
    // Grab the input tensor
    float * data_inputs = THCudaTensor_data(state, inputs);
    float * data_targets = THCudaTensor_data(state, targets);
    float * data_inside_weights = THCudaTensor_data(state, inside_weights);
    float * data_outside_weights = THCudaTensor_data(state, outside_weights);
    float * data_unsum_loss = THCudaTensor_data(state, unsumLoss);
    //float * output_flat = THCudaTensor_data(state, output);
    // compute sigma^2
    float sigma2 = sigma * sigma;
    cudaStream_t stream = THCState_getCurrentStream(state);
    WrapSmoothL1LossForwardLaucher(data_num, sigma2, size_average,
        data_inputs, data_targets,
        data_inside_weights, data_outside_weights,
        data_unsum_loss, stream);
    THCudaTensor_fill(state, output, THCudaTensor_sumall(state, unsumLoss));
    if (size_average)
        THCudaTensor_div(state, output, output, (float)data_num);
    THCudaTensor_free(state, unsumLoss);
    return 1;
}

int wrap_smooth_l1_loss_backward_cuda(float sigma, int size_average,
                   THCudaTensor * inputs, THCudaTensor * targets,
                   THCudaTensor * inside_weights, THCudaTensor * outside_weights,
                   THCudaTensor * grad_input1, THCudaTensor * grad_input2, THCudaTensor * grad_output)
{
    // check size
    if (!THCudaTensor_isSameSizeAs(state, inputs, targets))
        return 0;
    // data dim
    int data_dim = inputs->nDimension;
    int d=0, data_num=1;
    for (d=0; d < data_dim; d++)
    {
        data_num *= THCudaTensor_size(state, inputs, d);
    }
    // Grab the input tensor
    float * data_inputs = THCudaTensor_data(state, inputs);
    float * data_targets = THCudaTensor_data(state, targets);
    float * data_inside_weights = THCudaTensor_data(state, inside_weights);
    float * data_outside_weights = THCudaTensor_data(state, outside_weights);
    float * input_flat1 = THCudaTensor_data(state, grad_input1);
    float * input_flat2 = THCudaTensor_data(state, grad_input2);
    float * output_flat = THCudaTensor_data(state, grad_output);
    // compute sigma^2
    float sigma2 = sigma * sigma;
    cudaStream_t stream = THCState_getCurrentStream(state);

    WrapSmoothL1LossBackwardLaucher(data_num, sigma2, size_average,
        data_inputs, data_targets,
        data_inside_weights, data_outside_weights,
        input_flat1, input_flat2,
        output_flat, stream);

    return 1;
}
