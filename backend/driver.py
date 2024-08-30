import importlib.util
import sys
import tempfile
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import tvm.contrib.torch
from typing import TYPE_CHECKING

import pycuda.driver
pycuda.driver.init()

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
        gridX, gridY, gridZ, stream, tvm_module,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args
    ):
        tvm_module(*args)

_TVM_COUNTER = 0

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

    @staticmethod
    def get_tvm_target():
        dev = pycuda.driver.Device(0)
        capability_major = dev.get_attribute(pycuda.driver.device_attribute.COMPUTE_CAPABILITY_MAJOR)
        capability_minor = dev.get_attribute(pycuda.driver.device_attribute.COMPUTE_CAPABILITY_MINOR)
        arch = f"sm_{capability_major}{capability_minor}"
        max_threads_per_block = dev.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK)
        max_num_threads = dev.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
        thread_warp_size = dev.get_attribute(pycuda.driver.device_attribute.WARP_SIZE)
        max_shared_memory_per_block = dev.get_attribute(pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        registers_per_block = dev.get_attribute(pycuda.driver.device_attribute.MAX_REGISTERS_PER_BLOCK)
        return tvm.target.Target(f"cuda -arch={arch} -max_threads_per_block={max_threads_per_block} -max_num_threads={max_num_threads} -thread_warp_size={thread_warp_size} -max_shared_memory_per_block={max_shared_memory_per_block} -registers_per_block={registers_per_block}", host="llvm")

    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        tvmscript = kernel_asm.decode("utf-8")
        print("Building TVM module from script:")
        print(tvmscript)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.py') as ftvmscript:
            ftvmscript.write(tvmscript)
            ftvmscript.flush()
            global _TVM_COUNTER
            mod_name = f"__triton_tvm_module_{_TVM_COUNTER}"
            _TVM_COUNTER += 1
            spec = importlib.util.spec_from_file_location(mod_name, ftvmscript.name)
            assert spec is not None, "Failed to create spec from file location"
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            prim_func = mod.Module[name]
            wrapped = tvm.contrib.torch.as_torch(prim_func)
            target = TVMUtils.get_tvm_target()
            wrapped.build(target=target)
        return (
          mod_name,   # module
          wrapped,    # TVM PackedFunc
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
