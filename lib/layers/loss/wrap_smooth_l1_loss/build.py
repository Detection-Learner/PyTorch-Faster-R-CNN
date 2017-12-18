import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/wrap_smooth_l1_loss.c']
headers = ['src/wrap_smooth_l1_loss.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/wrap_smooth_l1_loss_cuda.c']
    headers += ['src/wrap_smooth_l1_loss_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print('\033[92m' + this_file + '\033[0m')
extra_objects = ['src/cuda/wrap_smooth_l1_loss_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.wrap_smooth_l1_loss',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
