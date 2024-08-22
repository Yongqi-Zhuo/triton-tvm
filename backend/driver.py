from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dims_stealer import patch_kernel_interface
else:
    # Use full path because Triton loads this module dynamically.
    from triton.backends.triton_tvm.dims_stealer import patch_kernel_interface


# This is no-op.
class TVMLauncher(object):
    def __init__(self, src, metadata):
        pass

    def __call__(
        self,
        gridX, gridY, gridZ, stream, cu_function,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args
    ):
        print("TVM Launcher called.")
        print("=== Source ===")
        print(cu_function.decode("utf-8"))
        print("===")

class TVMUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(TVMUtils, cls).__new__(cls)
        return cls.instance

    # Dummy.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    # Dummy.
    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (
          None,       # module
          kernel_asm, # function
          None,       # n_regs
          None        # n_spills
        )

class TVMDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = TVMUtils()
        self.launcher_cls = TVMLauncher
        self.binary_ext = "tvmir"
        patch_kernel_interface()

    # Remember to use triton.runtime.driver.set_active(TVMDriver())
    @staticmethod
    def is_active():
        return False

    def get_device_capability(self):
        return ("tvm", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        return "tvm"

    def set_current_device(self, device):
        assert device == "tvm"
        return

    def get_current_target(self):
        return GPUTarget("tvm", 0, 0)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
