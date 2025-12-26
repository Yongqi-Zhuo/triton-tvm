import tvm
import tvm.script
from tvm.script import tir as T

@tvm.script.ir_module
class Module:
        @T.prim_func
        def add_kernel(arg0: T.Buffer([T.int32(98432), ], "float32"), arg1: T.Buffer([T.int32(98432), ], "float32"), arg2: T.Buffer([T.int32(98432), ], "float32"), ):
                T.func_attr({"tir.noalias": True})
                alloc3 = T.alloc_buffer([T.int32(97), T.int32(1024), ], dtype="float32", scope="shared")
                alloc4 = T.alloc_buffer([T.int32(97), T.int32(1024), ], dtype="float32", scope="shared")
                for index5 in T.thread_binding(T.int32(0), T.int32(97), thread="blockIdx.x"):
                        for index8 in T.unroll(T.int32(0), T.int32(2)):
                                for index10 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        for index12 in T.vectorized(T.int32(0), T.int32(4)):
                                                with T.block():
                                                        axis14 = T.axis.spatial(T.int32(97), index5)
                                                        axis15 = T.axis.spatial(T.int32(1024), index12 + index10 * T.int32(4) + index8 * T.int32(512))
                                                        T.reads([arg0[axis14 * T.int32(1024) + axis15], ])
                                                        T.writes([alloc4[axis14, axis15], ])
                                                        alloc4[axis14, axis15] = T.if_then_else(axis14 * T.int32(1024) + axis15 < T.int32(98432), arg0[axis14 * T.int32(1024) + axis15], T.float32(0.00))
                        for index30 in T.unroll(T.int32(0), T.int32(2)):
                                for index31 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        for index32 in T.vectorized(T.int32(0), T.int32(4)):
                                                with T.block():
                                                        axis33 = T.axis.spatial(T.int32(97), index5)
                                                        axis34 = T.axis.spatial(T.int32(1024), index32 + index31 * T.int32(4) + index30 * T.int32(512))
                                                        T.reads([arg1[axis33 * T.int32(1024) + axis34], ])
                                                        T.writes([alloc3[axis33, axis34], ])
                                                        alloc3[axis33, axis34] = T.if_then_else(axis33 * T.int32(1024) + axis34 < T.int32(98432), arg1[axis33 * T.int32(1024) + axis34], T.float32(0.00))
                        for index45 in T.unroll(T.int32(0), T.int32(2)):
                                for index46 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        for index47 in T.vectorized(T.int32(0), T.int32(4)):
                                                with T.block():
                                                        axis48 = T.axis.spatial(T.int32(97), index5)
                                                        axis49 = T.axis.spatial(T.int32(1024), index47 + index46 * T.int32(4) + index45 * T.int32(512))
                                                        T.where(index5 * T.int32(1024) + (index47 + index46 * T.int32(4) + index45 * T.int32(512)) < T.int32(98432))
                                                        T.writes([arg2[axis48 * T.int32(1024) + axis49], ])
                                                        T.reads([alloc4[axis48, axis49], alloc3[axis48, axis49], ])
                                                        arg2[axis48 * T.int32(1024) + axis49] = alloc4[axis48, axis49] + alloc3[axis48, axis49]