from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class StolenData:
    grid: tuple[int, int, int]
    shapes: list[list[int]]
    strides: list[list[int]]

_STOLEN_DATA: StolenData | None = None
# This is a hack. We have to obtain the static gridDim and tensor shapes. So we patch kernel[grid].
def _dim_stealer(self, grid):
    def _patched_run(*args, **kwargs):
        global _STOLEN_DATA
        assert _STOLEN_DATA is None, "There is a pending kernel to be compiled. Do not call another kernel."
        assert grid is not None, "The grid is not specified. TVM backend requires the grid to be specified."
        next_grid = grid
        if callable(next_grid):
            next_grid = next_grid(kwargs)
        grid_size = len(next_grid)
        grid_0: int = next_grid[0]
        grid_1: int = next_grid[1] if grid_size > 1 else 1
        grid_2: int = next_grid[2] if grid_size > 2 else 1
        tensors = [t for t in args if torch.is_tensor(t)]
        shapes = [list(t.shape) for t in tensors]
        strides = [list(t.stride()) for t in tensors]
        _STOLEN_DATA = StolenData(
            grid=(grid_0, grid_1, grid_2),
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
    return data
