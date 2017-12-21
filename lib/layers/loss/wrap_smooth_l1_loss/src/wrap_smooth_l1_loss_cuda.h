/*
 * PyTorch Faster R-CNN
 * Written by Hongyu Pan, Yuanshun Cui
 *
 */

int wrap_smooth_l1_loss_forward_cuda(float sigma, int number,
                       THCudaTensor * inputs, THCudaTensor * targets,
                       THCudaTensor * inside_weights, THCudaTensor * outside_weights, THCudaTensor * output);

int wrap_smooth_l1_loss_backward_cuda(float sigma, int number,
                      THCudaTensor * inputs, THCudaTensor * targets,
                      THCudaTensor * inside_weights, THCudaTensor * outside_weights,
                      THCudaTensor * grad_input1, THCudaTensor * grad_input2, THCudaTensor * grad_output);
