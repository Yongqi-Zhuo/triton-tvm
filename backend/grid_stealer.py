_STOLEN_GRID = None
# This is a hack. We have to obtain the static gridDim. So we patch kernel[grid].
def _grid_stealer(self, grid):
    def _patched_run(*args, **kwargs):
        global _STOLEN_GRID
        assert _STOLEN_GRID is None, "There is a pending kernel to be compiled. Do not call another kernel."
        assert grid is not None, "The grid is not specified. TVM backend requires the grid to be specified."
        next_grid = grid
        if callable(next_grid):
            next_grid = next_grid(kwargs)
        grid_size = len(next_grid)
        grid_0: int = next_grid[0]
        grid_1: int = next_grid[1] if grid_size > 1 else 1
        grid_2: int = next_grid[2] if grid_size > 2 else 1
        _STOLEN_GRID = (grid_0, grid_1, grid_2)
        return self.run(grid=grid, warmup=False, *args, **kwargs)
    return _patched_run
_KERNEL_INTERFACE_PATCHED = False

def patch_kernel_interface():
    # Patch.
    global _KERNEL_INTERFACE_PATCHED
    if _KERNEL_INTERFACE_PATCHED is False:
        # Register the grid stealer.
        from triton.runtime.jit import KernelInterface
        KernelInterface.__getitem__ = _grid_stealer
        _KERNEL_INTERFACE_PATCHED = True

def retrieve_stolen_grid() -> tuple[int, int, int]:
    global _STOLEN_GRID
    assert _STOLEN_GRID is not None, "No valid grid is provided."
    grid = _STOLEN_GRID
    _STOLEN_GRID = None
    return grid
