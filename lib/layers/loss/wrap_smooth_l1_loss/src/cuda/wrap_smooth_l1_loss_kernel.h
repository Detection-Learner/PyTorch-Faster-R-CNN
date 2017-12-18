/*
 *  PyTorch Faster R-CNN
 *  Written by Yuanshun Cui
*/
#ifndef _WARP_SMOOTH_L1_LOSS_KERNEL
#define _WARP_SMOOTH_L1_LOSS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int WrapSmoothL1LossForwardLaucher(const int data_num, const float sigma2, const int size_average,
                                const float * data_inputs, const float * data_targets,
                                const float * data_inside_weights, const float * data_outside_weights,
                                float * output_flat, cudaStream_t stream);
int WrapSmoothL1LossBackwardLaucher(const int data_num, const float sigma2, const int size_average,
                                const float * data_inputs, const float * data_targets,
                                const float * data_inside_weights, const float * data_outside_weights,
                                float * input_flat1, float * input_flat2,
                                const float * output_flat, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
