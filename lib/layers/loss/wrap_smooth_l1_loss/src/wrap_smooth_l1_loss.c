/*
 * PyTorch Faster R-CNN
 * Written by Hongyu Pan, Yuanshun Cui
 *
 */

#include <TH/TH.h>
#include <math.h>

int wrap_smooth_l1_loss_forward(float sigma, int size_average,
                      THFloatTensor * inputs, THFloatTensor * targets,
                      THFloatTensor * inside_weights, THFloatTensor * outside_weights, THFloatTensor * output)
{
    // check size
    if (!THFloatTensor_isSameSizeAs(inputs, targets))
        return 0;
    // batch size
    int batch_size = THFloatTensor_size(inputs, 0);
    // data dim
    int data_dim = inputs->nDimension;//THFloatTensor_size(inputs, 1);
    int d=1, data_left_num=1;
    for (d=1; d < data_dim; d++)
    {
        data_left_num *= THFloatTensor_size(inputs, d);
    }
    // Grab the input tensor
    float * data_inputs = THFloatTensor_data(inputs);
    float * data_targets = THFloatTensor_data(targets);
    float * data_inside_weights = THFloatTensor_data(inside_weights);
    float * data_outside_weights = THFloatTensor_data(outside_weights);
    float * output_flat = THFloatTensor_data(output);
    // compute sigma^2
    float sigma2 = sigma * sigma;
    int i,j;
    // f(x) = w_out * 0.5 * (w_in * x)^2 * sigma2,  if |x| < 1
    //      = w_out * (|w_in * x| - 0.5 / sigma2),  else
    *output_flat = 0;
    for (i=0; i<batch_size; i++)
    {
        for (j=0; j<data_left_num;j++)
        {
            float val = data_inside_weights[i*data_left_num + j] * (data_inputs[i*data_left_num + j] - data_targets[i*data_left_num + j]);
            float abs_val = abs(val);
            if (abs_val < 1.0 / sigma2)
            {
                *output_flat += data_outside_weights[i*data_left_num + j] * 0.5 * val * val * sigma2;
            }
            else
            {
                *output_flat += data_outside_weights[i*data_left_num + j] * (abs_val - 0.5 / sigma2);
            }
        }
    }
    if (size_average)
    {
        *output_flat /= (float)(batch_size * data_left_num);
    }
    return 1;
}

int wrap_smooth_l1_loss_backward(float sigma, int size_average,
                      THFloatTensor * inputs, THFloatTensor * targets,
                      THFloatTensor * inside_weights, THFloatTensor * outside_weights,
                      THFloatTensor * grad_input1, THFloatTensor * grad_input2, THFloatTensor * grad_output)
{
    // check size
    if (!THFloatTensor_isSameSizeAs(inputs, targets))
        return 0;
    // batch size
    int batch_size = THFloatTensor_size(inputs, 0);
    // data dim
    int data_dim = inputs->nDimension;
    // Grab the input tensor
    float * data_inputs = THFloatTensor_data(inputs);
    float * data_targets = THFloatTensor_data(targets);
    float * data_inside_weights = THFloatTensor_data(inside_weights);
    float * data_outside_weights = THFloatTensor_data(outside_weights);
    float * input_flat1 = THFloatTensor_data(grad_input1);
    float * input_flat2 = THFloatTensor_data(grad_input2);
    float * output_flat = THFloatTensor_data(grad_output);
    // compute sigma^2
    float sigma2 = sigma * sigma;
    int i, j, d, data_left_num = 1;
    for (d=1; d < data_dim; d++)
    {
        data_left_num *= THFloatTensor_size(inputs, d);
    }
    // f'(x) = w_out * w_in * sigma2 * (w_in * x),  if |x|<1
    //       = w_out * w_in *  sign(w_in * x),  else
    for (i=0; i<batch_size; i++)
    {
        for (j=0; j<data_left_num;j++)
        {
            float val = data_inside_weights[i*data_left_num + j] * (data_inputs[i*data_left_num + j] - data_targets[i*data_left_num + j]);
            float abs_val = abs(val);
            if (abs_val < 1.0 / sigma2)
            {
                input_flat1[i*data_left_num + j] = data_outside_weights[i*data_left_num + j] * data_inside_weights[i*data_left_num + j] * val * sigma2;
            }
            else
            {
                input_flat1[i*data_left_num + j] = data_outside_weights[i*data_left_num + j] * data_inside_weights[i*data_left_num + j] * ((0<val) - (val<0));
            }
            input_flat2[i*data_left_num + j] = -input_flat1[i*data_left_num + j] * output_flat[0] / (size_average ? batch_size*data_left_num : 1);
            input_flat1[i*data_left_num + j] = input_flat1[i*data_left_num + j] * output_flat[0] / (size_average ? batch_size*data_left_num : 1);
        }
    }
    return 1;
}
