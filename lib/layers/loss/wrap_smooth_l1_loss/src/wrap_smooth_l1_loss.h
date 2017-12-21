/*
 * PyTorch Faster R-CNN
 * Written by Hongyu Pan, Yuanshun Cui
 *
 */

int wrap_smooth_l1_loss_forward(float sigma, int number,
                       THFloatTensor * inputs, THFloatTensor * targets,
                       THFloatTensor * inside_weights, THFloatTensor * outside_weights, THFloatTensor * output);

int wrap_smooth_l1_loss_backward(float sigma, int number,
                      THFloatTensor * inputs, THFloatTensor * targets,
                      THFloatTensor * inside_weights, THFloatTensor * outside_weights,
                      THFloatTensor * grad_input1, THFloatTensor * grad_input2, THFloatTensor * grad_output);
