#-----------------------
# PyTorch Faster R-CNN
# Written by Hongyu Pan
#-----------------------
from torch.utils.ffi import _wrap_function
from ._my_smooth_l1_loss import lob as _lib, ffi as _ffi

__all__ = []

def _import_symbols(locals):
    for symbol im dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)

_import_symbols(locals())
