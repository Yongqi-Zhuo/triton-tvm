import tvm
import tvm.script
from tvm.script import tir as T

@tvm.script.ir_module
class Module:
        @T.prim_func
        def softmax_kernel(arg0: T.Buffer([T.int32(1823), T.int32(781), ], "float32"), arg1: T.Buffer([T.int32(1823), T.int32(781), ], "float32"), ):
                T.func_attr({"tir.noalias": True})
                alloc2 = T.alloc_buffer([T.int32(1823), ], dtype="float32", scope="shared")
                alloc3 = T.alloc_buffer([T.int32(1823), T.int32(1024), ], dtype="float32", scope="shared")
                alloc4 = T.alloc_buffer([T.int32(1823), ], dtype="float32", scope="shared")
                alloc5 = T.alloc_buffer([T.int32(1823), T.int32(1024), ], dtype="float32", scope="shared")
                for index6 in T.thread_binding(T.int32(0), T.int32(1823), thread="blockIdx.x"):
                        for index9 in T.unroll(T.int32(0), T.int32(8)):
                                for index11 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        with T.block():
                                                axis13 = T.axis.spatial(T.int32(1823), index6)
                                                axis14 = T.axis.spatial(T.int32(1024), index11 + index9 * T.int32(128))
                                                T.reads([arg1[T.truncdiv(axis13 * T.int32(781) + axis14, T.int32(781)), (axis13 * T.int32(781) + axis14) % T.int32(781)], ])
                                                T.writes([alloc5[axis13, axis14], ])
                                                alloc5[axis13, axis14] = T.if_then_else(axis14 < T.int32(781), arg1[T.truncdiv(axis13 * T.int32(781) + axis14, T.int32(781)), (axis13 * T.int32(781) + axis14) % T.int32(781)], T.min_value("float32"))
                        for index28 in T.serial(T.int32(0), T.int32(8)):
                                for index29 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        with T.block():
                                                axis30 = T.axis.spatial(T.int32(1823), index6)
                                                axis31 = T.axis.reduce(T.int32(1024), index29 + index28 * T.int32(128))
                                                T.writes([alloc4[axis30], ])
                                                with T.init():
                                                        alloc4[axis30] = T.min_value("float32")
                                                T.reads([alloc5[axis30, axis31], ])
                                                alloc4[axis30] = T.max(alloc4[axis30], alloc5[axis30, axis31])
                        for index38 in T.unroll(T.int32(0), T.int32(8)):
                                for index39 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        with T.block():
                                                axis40 = T.axis.spatial(T.int32(1823), index6)
                                                axis41 = T.axis.spatial(T.int32(1024), index39 + index38 * T.int32(128))
                                                T.writes([alloc3[axis40, axis41], ])
                                                T.reads([alloc4[axis40], alloc5[axis40, axis41], ])
                                                alloc3[axis40, axis41] = T.exp(alloc5[axis40, axis41] - alloc4[axis40])
                        for index49 in T.serial(T.int32(0), T.int32(8)):
                                for index50 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        with T.block():
                                                axis51 = T.axis.spatial(T.int32(1823), index6)
                                                axis52 = T.axis.reduce(T.int32(1024), index50 + index49 * T.int32(128))
                                                T.writes([alloc2[axis51], ])
                                                with T.init():
                                                        alloc2[axis51] = T.float32(0.00)
                                                T.reads([alloc3[axis51, axis52], ])
                                                alloc2[axis51] = alloc2[axis51] + alloc3[axis51, axis52]
                        for index59 in T.unroll(T.int32(0), T.int32(8)):
                                for index60 in T.thread_binding(T.int32(0), T.int32(128), thread="threadIdx.x"):
                                        with T.block():
                                                axis61 = T.axis.spatial(T.int32(1823), index6)
                                                axis62 = T.axis.spatial(T.int32(1024), index60 + index59 * T.int32(128))
                                                T.where(index60 + index59 * T.int32(128) < T.int32(781))
                                                T.writes([arg0[T.truncdiv(axis61 * T.int32(781) + axis62, T.int32(781)), (axis61 * T.int32(781) + axis62) % T.int32(781)], ])
                                                T.reads([alloc4[axis61], alloc2[axis61], alloc5[axis61, axis62], ])
                                                arg0[T.truncdiv(axis61 * T.int32(781) + axis62, T.int32(781)), (axis61 * T.int32(781) + axis62) % T.int32(781)] = T.exp(alloc5[axis61, axis62] - alloc4[axis61]) / alloc2[axis61]