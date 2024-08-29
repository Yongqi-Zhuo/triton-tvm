from collections import defaultdict
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class StolenData:
    owner: int
    grid: list[int]
    shapes: list[list[int]]
    strides: list[list[int]]

_HAS_BEEN_STOLEN = defaultdict(bool)
_STOLEN_DATA: StolenData | None = None
# This is a hack. We have to obtain the static gridDim and tensor shapes. So we patch kernel[grid].
def _dim_stealer(self, grid):
    def _patched_run(*args, **kwargs):
        if _HAS_BEEN_STOLEN[id(self)]:
            # This is already compiled.
            return self.run(grid=grid, warmup=False, *args, **kwargs)
        global _STOLEN_DATA
        if _STOLEN_DATA is not None:
            # Wow, it seems our compilation result is cached. Let's use it.
            _STOLEN_DATA = None
            return self.run(grid=grid, warmup=False, *args, **kwargs)
        # assert _STOLEN_DATA is None, "There is a pending kernel to be compiled. Do not call another kernel."
        assert grid is not None, "The grid is not specified. TVM backend requires the grid to be specified."
        next_grid = grid
        if callable(next_grid):
            next_grid = next_grid(kwargs)
        tensors = [t for t in args if torch.is_tensor(t)]
        shapes = [list(t.shape) for t in tensors]
        strides = [list(t.stride()) for t in tensors]
        _STOLEN_DATA = StolenData(
            owner=id(self),
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
    global _STOLEN_DATA
    assert _STOLEN_DATA is not None, "No valid data is provided."
    data = _STOLEN_DATA
    _STOLEN_DATA = None
    _HAS_BEEN_STOLEN[data.owner] = True
    return data
