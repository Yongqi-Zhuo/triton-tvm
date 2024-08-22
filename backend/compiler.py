from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, nvidia, triton_tvm
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import hashlib
import functools

if TYPE_CHECKING:
    from dims_stealer import retrieve_stolen_dims
else:
    # Use full path because Triton loads this module dynamically.
    from triton.backends.triton_tvm.dims_stealer import retrieve_stolen_dims


@dataclass(frozen=True)
class TVMOptions:
    debug: bool = False
    arch: str | int = 0
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: tuple[str] = ("ieee", )

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class TVMBackend(BaseBackend):
    binary_ext = 'tvmir'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'tvm'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = 86

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in TVMOptions.__dataclass_fields__.keys() if k in opts})
        return TVMOptions(**args)

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        # if capability // 10 >= 8:
        #     passes.ttgpuir.add_f32_dot_tc(pm)
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.common.add_cse(pm)
        # if capability // 10 >= 8:
        #     passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # passes.ttgpuir.add_pipeline(pm, opt.num_stages)
        # passes.ttgpuir.add_prefetch(pm)
        # passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        # passes.ttgpuir.add_reduce_data_duplication(pm)
        # passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        # if capability // 10 >= 9:
        #     nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
        #     nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_tvmir(mod, metadata, opt):
        metadata["name"] = "kernel_main"
        stolen_dims = retrieve_stolen_dims()
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        triton_tvm.passes.ttgpuir.add_convert_to_tvm(pm, stolen_dims.grid, stolen_dims.shapes, stolen_dims.strides)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["tvmir"] = lambda src, metadata: self.make_tvmir(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target
