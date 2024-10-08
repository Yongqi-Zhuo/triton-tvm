from collections import defaultdict
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class StolenData:
    grid: list[int]
    shapes: list[list[int]]
    strides: list[list[int]]

_STOLEN_DATA: StolenData | None = None
# This is a hack. We have to obtain the static gridDim and tensor shapes. So we patch kernel[grid].
def _dim_stealer(self, grid):
    def _patched_run(*args, **kwargs):
        global _STOLEN_DATA
        assert grid is not None, "The grid is not specified. TVM backend requires the grid to be specified."
        next_grid = grid
        if callable(next_grid):
            next_grid = next_grid(kwargs)
        tensors = [t for t in args if torch.is_tensor(t)]
        shapes = [list(t.shape) for t in tensors]
        strides = [list(t.stride()) for t in tensors]
        _STOLEN_DATA = StolenData(
            grid=next_grid,
            shapes=shapes,
            strides=strides,
        )
        return self.run(grid=grid, warmup=False, *args, **kwargs)
    return _patched_run
_KERNEL_INTERFACE_PATCHED = False

def patch_kernel_interface():
    # Patch.
    global _KERNEL_INTERFACE_PATCHED
    if _KERNEL_INTERFACE_PATCHED is False:
        # Register the dim stealer.
        from triton.runtime.jit import KernelInterface
        KernelInterface.__getitem__ = _dim_stealer
        _KERNEL_INTERFACE_PATCHED = True

def retrieve_stolen_dims() -> StolenData:
    assert _STOLEN_DATA is not None, "No valid data is provided."
    return _STOLEN_DATA
