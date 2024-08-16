import os
import sys
import torch
import triton
import triton.testing
import tvm
import tvm.contrib.torch
import tvm.meta_schedule as ms
import tvm.relax as relax
import tvm.script
from tvm.script import tir as T
from tvm.script import relax as R
from typing import Literal

from code_triton import softmax as softmax_triton

# BLOCK_SIZE = 1024
# N_ROWS = 1823
# N_COLS = 781
BLOCK_SIZE = 2048
N_ROWS = 2560
N_COLS = 1989
# BLOCK_SIZE = 256
# N_ROWS = 256
# N_COLS = 256
assert N_COLS <= BLOCK_SIZE

DEVICE: Literal['cuda'] | Literal['cpu'] = 'cuda'
def get_tvm_target() -> tvm.target.Target:
    match DEVICE:
        case 'cuda':
            return tvm.target.Target("cuda -arch=sm_86 -max_threads_per_block=1024 -max_num_threads=1536 -thread_warp_size=32 -max_shared_memory_per_block=49152 -registers_per_block=65536", host="llvm -mtriple=x86_64-pc-linux-gnu -num-cores=16")
        case 'cpu':
            return tvm.target.Target("llvm -mtriple=x86_64-pc-linux-gnu -num-cores=16")
        case _:
            raise ValueError(f"Unsupported device: {DEVICE}")


# See: tests/python/tir-transform/test_tir_transform_lower_cross_thread_reduction.py in github.com/apache/tvm

# Works. However MetaSchedule cannot tune this implementation.
@T.prim_func
def softmax_tvm_impl_official(var_A: T.handle, var_T_softmax_norm: T.handle) -> None:
    T.func_attr({"tir.noalias": True})
    # Here, we require N_COLS == BLOCK_SIZE and is multiple of 32
    A = T.match_buffer(var_A, [N_ROWS, N_COLS], dtype="float32")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, [N_ROWS, N_COLS], dtype="float32")
    T_softmax_maxelem_shared = T.alloc_buffer([N_ROWS], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([N_ROWS], dtype="float32", scope="shared")
    for i0 in T.thread_binding(0, N_ROWS, thread="blockIdx.x"):
        for ax0_0 in T.serial(0, BLOCK_SIZE // 32):
            for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem"):
                    i0_1 = T.axis.spatial(N_ROWS, i0)
                    k = T.axis.reduce(BLOCK_SIZE, ax0_0 * 32 + ax0_1)
                    T.reads([A[i0_1, k]])
                    T.writes([T_softmax_maxelem_shared[i0_1]])
                    with T.init():
                        T_softmax_maxelem_shared[i0_1] = T.min_value("float32")
                    T_softmax_maxelem_shared[i0_1] = T.max(
                        T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                    )
        for ax0_0 in T.serial(0, BLOCK_SIZE // 32):
            for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_expsum"):
                    i0_2 = T.axis.spatial(N_ROWS, i0)
                    k = T.axis.reduce(BLOCK_SIZE, ax0_0 * 32 + ax0_1)
                    T.reads(
                        [
                            A[i0_2, k],
                            T_softmax_maxelem_shared[i0_2],
                        ]
                    )
                    T.writes([T_softmax_expsum_shared[i0_2]])
                    with T.init():
                        T_softmax_expsum_shared[i0_2] = T.float32(0)
                    T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                        A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                    )
        for i1_0 in T.serial(0, BLOCK_SIZE // 32):
            for i1_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_3 = T.axis.spatial(N_ROWS, i0)
                    i1 = T.axis.spatial(BLOCK_SIZE, i1_0 * 32 + i1_1)
                    T.reads(
                        [
                            A[i0_3, i1],
                            T_softmax_maxelem_shared[i0_3],
                            T_softmax_expsum_shared[i0_3],
                        ]
                    )
                    T.writes([T_softmax_norm[i0_3, i1]])
                    T.block_attr({"axis": 1})
                    T_softmax_norm[i0_3, i1] = (
                        T.exp(
                            A[i0_3, i1] - T_softmax_maxelem_shared[i0_3],
                            dtype="float32",
                        )
                        / T_softmax_expsum_shared[i0_3]
                    )

# MetaSchedule failed to find a schedule for the following implementation.
@T.prim_func
def softmax_tvm_impl_official_unscheduled(var_A: T.handle, var_T_softmax_norm: T.handle) -> None:
    T.func_attr({"tir.noalias": True})
    # Here, we require N_COLS == BLOCK_SIZE and is multiple of 32
    A = T.match_buffer(var_A, [N_ROWS, N_COLS], dtype="float32")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, [N_ROWS, N_COLS], dtype="float32")
    T_softmax_maxelem_shared = T.alloc_buffer([N_ROWS], dtype="float32")
    T_softmax_expsum_shared = T.alloc_buffer([N_ROWS], dtype="float32")
    for i0 in T.grid(N_ROWS):
        for ax0 in T.grid(BLOCK_SIZE):
            with T.block("T_softmax_maxelem"):
                i0_1 = T.axis.spatial(N_ROWS, i0)
                k = T.axis.reduce(BLOCK_SIZE, ax0)
                T.reads([A[i0_1, k]])
                T.writes([T_softmax_maxelem_shared[i0_1]])
                with T.init():
                    T_softmax_maxelem_shared[i0_1] = T.min_value("float32")
                T_softmax_maxelem_shared[i0_1] = T.max(
                    T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                )
        for ax0 in T.grid(BLOCK_SIZE):
            with T.block("T_softmax_expsum"):
                i0_2 = T.axis.spatial(N_ROWS, i0)
                k = T.axis.reduce(BLOCK_SIZE, ax0)
                T.reads(
                    [
                        A[i0_2, k],
                        T_softmax_maxelem_shared[i0_2],
                    ]
                )
                T.writes([T_softmax_expsum_shared[i0_2]])
                with T.init():
                    T_softmax_expsum_shared[i0_2] = T.float32(0)
                T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                )
        for ax0 in T.grid(BLOCK_SIZE):
            with T.block("T_softmax_norm"):
                i0_3 = T.axis.spatial(N_ROWS, i0)
                i1 = T.axis.spatial(BLOCK_SIZE, ax0)
                T.reads(
                    [
                        A[i0_3, i1],
                        T_softmax_maxelem_shared[i0_3],
                        T_softmax_expsum_shared[i0_3],
                    ]
                )
                T.writes([T_softmax_norm[i0_3, i1]])
                T_softmax_norm[i0_3, i1] = (
                    T.exp(
                        A[i0_3, i1] - T_softmax_maxelem_shared[i0_3],
                        dtype="float32",
                    )
                    / T_softmax_expsum_shared[i0_3]
                )

# This will not work, due to MetaSchedule not being able to tune.
@T.prim_func
def softmax_tvm_impl_transpiled(input_ptr: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), "float32"), output_ptr: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), "float32")):
    T.func_attr({"tir.noalias": True})
    # Dimensions of any of the Triton tensors require num_programs (i.e., N_ROWS)
    row = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32")
    row_max = T.alloc_buffer([T.int32(N_ROWS)], dtype="float32")
    row_minus_max = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32")
    numerator = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32")
    denominator = T.alloc_buffer([T.int32(N_ROWS)], dtype="float32")
    softmax_output = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32")
    for program_id in T.grid(T.int32(N_ROWS)):
        for col_offsets in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("load_row"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_col_offsets = T.axis.spatial(T.int32(BLOCK_SIZE), col_offsets)
                T.reads([input_ptr[i_program_id, i_col_offsets]])
                T.writes([row[i_program_id, i_col_offsets]])
                row[i_program_id, i_col_offsets] = T.if_then_else(
                    i_col_offsets < N_COLS,
                    input_ptr[i_program_id, i_col_offsets],
                    T.float32(-float("inf")),
                )
        for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("row_max"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_axis_0 = T.axis.reduce(T.int32(BLOCK_SIZE), axis_0)
                T.reads([row[i_program_id, i_axis_0]])
                T.writes([row_max[i_program_id]])
                with T.init():
                    row_max[i_program_id] = T.float32(-float("inf"))
                row_max[i_program_id] = T.max(
                    row_max[i_program_id],
                    row[i_program_id, i_axis_0]
                )
        for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("row_minus_max"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0)
                T.reads([row[i_program_id, i_axis_0], row_max[i_program_id]])
                T.writes([row_minus_max[i_program_id, i_axis_0]])
                row_minus_max[i_program_id, i_axis_0] = row[i_program_id, i_axis_0] - row_max[i_program_id]
        for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("numerator"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0)
                T.reads([row_minus_max[i_program_id, i_axis_0]])
                T.writes([numerator[i_program_id, i_axis_0]])
                numerator[i_program_id, i_axis_0] = T.exp(
                    row_minus_max[i_program_id, i_axis_0],
                    dtype="float32",
                )
        for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("denominator"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_axis_0 = T.axis.reduce(T.int32(BLOCK_SIZE), axis_0)
                T.reads([numerator[i_program_id, i_axis_0]])
                T.writes([denominator[i_program_id]])
                with T.init():
                    denominator[i_program_id] = T.float32(0)
                denominator[i_program_id] = denominator[i_program_id] + numerator[i_program_id, i_axis_0]
        for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
            with T.block("softmax_output"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0)
                T.reads([numerator[i_program_id, i_axis_0], denominator[i_program_id]])
                T.writes([softmax_output[i_program_id, i_axis_0]])
                softmax_output[i_program_id, i_axis_0] = numerator[i_program_id, i_axis_0] / denominator[i_program_id]
        for _output_size in T.grid(T.int32(N_COLS)):
            with T.block("store_softmax_output"):
                i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                i__output_size = T.axis.spatial(T.int32(N_COLS), _output_size)
                T.reads([softmax_output[i_program_id, i__output_size]])
                T.writes([output_ptr[i_program_id, i__output_size]])
                output_ptr[i_program_id, i__output_size] = softmax_output[i_program_id, i__output_size]

WARP_SIZE = 32
NUM_WARPS = 4
NUM_THREADS = WARP_SIZE * NUM_WARPS
assert BLOCK_SIZE % WARP_SIZE == 0

# This works. Note that we have to convert from TritonGPU dialect.
@T.prim_func
def softmax_tvm_impl_transpiled_scheduled(input_ptr: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), "float32"), output_ptr: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), "float32")):
    T.func_attr({"tir.noalias": True})
    # Dimensions of any of the Triton tensors require num_programs (i.e., N_ROWS)
    row = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32", scope="shared")
    row_max = T.alloc_buffer([T.int32(N_ROWS)], dtype="float32", scope="shared")
    row_minus_max = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32", scope="shared")
    numerator = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32", scope="shared")
    denominator = T.alloc_buffer([T.int32(N_ROWS)], dtype="float32", scope="shared")
    softmax_output = T.alloc_buffer([T.int32(N_ROWS), T.int32(BLOCK_SIZE)], dtype="float32", scope="shared")
    for program_id in T.thread_binding(T.int32(0), T.int32(N_ROWS), thread="blockIdx.x"):
        for col_offsets_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for col_offsets_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("load_row"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_col_offsets = T.axis.spatial(T.int32(BLOCK_SIZE), col_offsets_0 * T.int32(NUM_THREADS) + col_offsets_1)
                    T.reads([input_ptr[i_program_id, i_col_offsets]])
                    T.writes([row[i_program_id, i_col_offsets]])
                    row[i_program_id, i_col_offsets] = T.if_then_else(
                        i_col_offsets < N_COLS,
                        input_ptr[i_program_id, i_col_offsets],
                        # Here, min_value is required by LowerCrossThreadReduction pass
                        # T.float32(-float("inf")),
                        T.min_value("float32"),
                    )
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("row_max"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.reduce(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([row[i_program_id, i_axis_0]])
                    T.writes([row_max[i_program_id]])
                    with T.init():
                        row_max[i_program_id] = T.min_value("float32")
                    row_max[i_program_id] = T.max(
                        row_max[i_program_id],
                        row[i_program_id, i_axis_0]
                    )
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("row_minus_max"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([row[i_program_id, i_axis_0], row_max[i_program_id]])
                    T.writes([row_minus_max[i_program_id, i_axis_0]])
                    row_minus_max[i_program_id, i_axis_0] = row[i_program_id, i_axis_0] - row_max[i_program_id]
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("numerator"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([row_minus_max[i_program_id, i_axis_0]])
                    T.writes([numerator[i_program_id, i_axis_0]])
                    numerator[i_program_id, i_axis_0] = T.exp(
                        row_minus_max[i_program_id, i_axis_0],
                        dtype="float32",
                    )
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("denominator"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.reduce(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([numerator[i_program_id, i_axis_0]])
                    T.writes([denominator[i_program_id]])
                    with T.init():
                        denominator[i_program_id] = T.float32(0)
                    denominator[i_program_id] = denominator[i_program_id] + numerator[i_program_id, i_axis_0]
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("softmax_output"):
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([numerator[i_program_id, i_axis_0], denominator[i_program_id]])
                    T.writes([softmax_output[i_program_id, i_axis_0]])
                    softmax_output[i_program_id, i_axis_0] = numerator[i_program_id, i_axis_0] / denominator[i_program_id]
        for axis_0_0 in T.serial(T.int32(0), T.int32(BLOCK_SIZE // NUM_THREADS)):
            for axis_0_1 in T.thread_binding(T.int32(0), T.int32(NUM_THREADS), thread="threadIdx.x"):
                with T.block("store_softmax_output"):
                    T.where(axis_0_0 * T.int32(NUM_THREADS) + axis_0_1 < N_COLS)
                    i_program_id = T.axis.spatial(T.int32(N_ROWS), program_id)
                    i_axis_0 = T.axis.spatial(T.int32(BLOCK_SIZE), axis_0_0 * T.int32(NUM_THREADS) + axis_0_1)
                    T.reads([softmax_output[i_program_id, i_axis_0]])
                    T.writes([output_ptr[i_program_id, i_axis_0]])
                    output_ptr[i_program_id, i_axis_0] = softmax_output[i_program_id, i_axis_0]

# This is not a single-kernel implementation, so will be slow.
@tvm.script.ir_module
class softmax_tvm_impl_multiple_prim_funcs:
    @T.prim_func(private=True)
    def load_row(inp_0: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), "float32"), result: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for col_offsets in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("load_row"):
                    i_program_id, i_col_offsets = T.axis.remap("SS", [program_id, col_offsets])
                    T.reads([inp_0[i_program_id, i_col_offsets]])
                    T.writes([result[i_program_id, i_col_offsets]])
                    result[i_program_id, i_col_offsets] = T.if_then_else(
                        i_col_offsets < N_COLS,
                        inp_0[i_program_id, i_col_offsets],
                        T.float32(-float("inf")),
                    )

    @T.prim_func(private=True)
    def compute_row_max(row: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("row_max"):
                    i_program_id, i_axis_0 = T.axis.remap("SR", [program_id, axis_0])
                    T.reads([row[i_program_id, i_axis_0]])
                    T.writes([result[i_program_id]])
                    with T.init():
                        result[i_program_id] = T.float32(-float("inf"))
                    result[i_program_id] = T.max(
                        result[i_program_id],
                        row[i_program_id, i_axis_0]
                    )

    @T.prim_func(private=True)
    def compute_row_minus_max(row: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), row_max: T.Buffer((T.int32(N_ROWS)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("row_minus_max"):
                    i_program_id, i_axis_0 = T.axis.remap("SS", [program_id, axis_0])
                    T.reads([row[i_program_id, i_axis_0], row_max[i_program_id]])
                    T.writes([result[i_program_id, i_axis_0]])
                    result[i_program_id, i_axis_0] = row[i_program_id, i_axis_0] - row_max[i_program_id]

    @T.prim_func(private=True)
    def compute_numerator(row_minus_max: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("numerator"):
                    i_program_id, i_axis_0 = T.axis.remap("SS", [program_id, axis_0])
                    T.reads([row_minus_max[i_program_id, i_axis_0]])
                    T.writes([result[i_program_id, i_axis_0]])
                    result[i_program_id, i_axis_0] = T.exp(
                        row_minus_max[i_program_id, i_axis_0],
                        dtype="float32",
                    )

    @T.prim_func(private=True)
    def compute_denominator(numerator: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("denominator"):
                    i_program_id, i_axis_0 = T.axis.remap("SR", [program_id, axis_0])
                    T.reads([numerator[i_program_id, i_axis_0]])
                    T.writes([result[i_program_id]])
                    with T.init():
                        result[i_program_id] = T.float32(0)
                    result[i_program_id] = result[i_program_id] + numerator[i_program_id, i_axis_0]

    @T.prim_func(private=True)
    def compute_softmax_output(numerator: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), denominator: T.Buffer((T.int32(N_ROWS)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for axis_0 in T.grid(T.int32(BLOCK_SIZE)):
                with T.block("softmax_output"):
                    i_program_id, i_axis_0 = T.axis.remap("SS", [program_id, axis_0])
                    T.reads([numerator[i_program_id, i_axis_0], denominator[i_program_id]])
                    T.writes([result[i_program_id, i_axis_0]])
                    result[i_program_id, i_axis_0] = numerator[i_program_id, i_axis_0] / denominator[i_program_id]

    @T.prim_func
    def store_softmax_output(softmax_output: T.Buffer((T.int32(N_ROWS), T.int32(BLOCK_SIZE)), dtype="float32"), result: T.Buffer((T.int32(N_ROWS), T.int32(N_COLS)), dtype="float32")):
        T.func_attr({"tir.noalias": True})
        for program_id in T.grid(T.int32(N_ROWS)):
            for _output_size in T.grid(T.int32(N_COLS)):
                with T.block("store_softmax_output"):
                    i_program_id, i__output_size = T.axis.remap("SS", [program_id, _output_size])
                    T.reads([softmax_output[i_program_id, i__output_size]])
                    T.writes([result[i_program_id, i__output_size]])
                    result[i_program_id, i__output_size] = softmax_output[i_program_id, i__output_size]

    @R.function
    def main(inp_0: R.Tensor((T.int32(N_ROWS), T.int32(N_COLS)), dtype="float32")) -> R.Tensor((T.int32(N_ROWS), T.int32(N_COLS)), dtype="float32"):
        cls = softmax_tvm_impl_multiple_prim_funcs
        with R.dataflow():
            row = R.call_tir(cls.load_row, (inp_0,), out_sinfo=R.Tensor((T.int32(N_ROWS), T.int32(BLOCK_SIZE),), dtype="float32"))
            row_max = R.call_tir(cls.compute_row_max, (row,), out_sinfo=R.Tensor((T.int32(N_ROWS),), dtype="float32"))
            row_minus_max = R.call_tir(cls.compute_row_minus_max, (row, row_max,), out_sinfo=R.Tensor((T.int32(N_ROWS), T.int32(BLOCK_SIZE),), dtype="float32"))
            numerator = R.call_tir(cls.compute_numerator, (row_minus_max,), out_sinfo=R.Tensor((T.int32(N_ROWS), T.int32(BLOCK_SIZE),), dtype="float32"))
            denominator = R.call_tir(cls.compute_denominator, (numerator,), out_sinfo=R.Tensor((T.int32(N_ROWS),), dtype="float32"))
            softmax_output = R.call_tir(cls.compute_softmax_output, (numerator, denominator,), out_sinfo=R.Tensor((T.int32(N_ROWS), T.int32(BLOCK_SIZE),), dtype="float32"))
            result = R.call_tir(cls.store_softmax_output, (softmax_output,), out_sinfo=R.Tensor((T.int32(N_ROWS), T.int32(N_COLS),), dtype="float32"))
            R.output(result)
        return result

softmax_tvm_kernel = tvm.contrib.torch.as_torch(softmax_tvm_impl_transpiled_scheduled)

def softmax_tvm(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    softmax_tvm_kernel(x, y)
    return y

def tune():
    print("Tuning...")
    target = get_tvm_target()
    softmax_tvm_kernel.tune(
        target=target,
        max_trials_global=64,
    )
    print("After tuning:")
    softmax_tvm_kernel.ir_module.show()

def build():
    print("Building softmax_tvm...")
    target = get_tvm_target()
    softmax_tvm_kernel.build(target=target)

def test_correctness():
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    x = torch.randn((N_ROWS, N_COLS), dtype=torch.float32, device=device, requires_grad=False)
    # print("x:", x)
    print("Computing with Torch...")
    y_torch = torch.nn.functional.softmax(x, dim=1)
    # print("y_torch:", y_torch)
    print("Computing with TVM...")
    y_tvm = softmax_tvm(x)
    # print("y_tvm:", y_tvm)
    assert torch.allclose(y_tvm, y_torch, rtol=1e-5, atol=1e-5)
    print("ok")

def test_benchmark():
    assert DEVICE == 'cuda'
    x = torch.randn((N_ROWS, N_COLS), dtype=torch.float32, device=DEVICE, requires_grad=False)
    print("=== Benchmarking softmax ===")
    print("Benchmarking torch.nn.functional.softmax... ", end="")
    t_torch: float = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1))
    print(t_torch)
    print("Benchmarking softmax_tvm... ", end="")
    t_tvm: float = triton.testing.do_bench(lambda: softmax_tvm(x))
    print(t_tvm)
    print("Benchmarking softmax_triton... ", end="")
    t_triton: float = triton.testing.do_bench(lambda: softmax_triton(x))
    print(t_triton)
    print("===")
    print("Speedup TVM:", t_torch / t_tvm)
    print("Speedup Triton:", t_torch / t_triton)
    print("===")

def main() -> int:
    # tune()
    build()
    test_correctness()
    test_benchmark()
    return 0

if __name__ == '__main__':
    sys.exit(main())
