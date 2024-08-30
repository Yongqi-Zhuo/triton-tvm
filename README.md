# TritonTVM

A transpiler from Triton to TVM.

## Design

### TVM Dialect

An MLIR dialect for TVM TensorIR is devised.

| TensorIR | TVM Dialect | Notes |
|----------|-------------|-------|
| `T.var()` | `tvm.var` | |
| `T.match_buffer` | `tvm.match_buffer` | |
| `T.alloc_buffer` | `tvm.alloc_buffer` | |
| `for k in range(n): ...` | `scf.for {tvm.for_kind = serial}` | `serial`, `parallel`, `vectorized`, `unroll`, `thread_binding` |
| `T.block` | `tvm.block` | |
| `T.where` | `tvm.where` | |
| `T.axis.spatial(k)` | `tvm.axis spatial %k` | `spatial`, `reduce`, `scan`, `opaque` |
| `A[i, j]` | `tvm.ref %A[%i, %j]` | |
| `T.if_then_else` | `tvm.if_then_else` | |
| `T.reads` | `tvm.read` | |
| `T.writes` | `tvm.write` | |
| `A[i, j] = ...` | `tvm.assign %ref = %value` | |
| `T.init` | `tvm.init` | |

### Pointer Analysis

Refer to `triton-shared` and `triton-linalg`.

### Conversion

Basically, by pattern matching.

| Triton Dialect | TVM Dialect |
|--------|-----|
| `tt.load` | `tvm.block` + `tvm.if_then_else` |
| `tt.store` | `tvm.block` + `tvm.where` |
| `tt.reduce` | Materialize with `tvm.block`, using implicit reduction axes. |
| `arith.*` | Do NOT materialze. Keep in the form. |
| `for k: ...` | `scf.for {tvm.for_kind = serial}` |
| `for k: ptrs += stride` | `ptrs = offset + k * stride` |
| `for k: acc += ...` | Materialize with `tvm.block`, with `tvm.axis reduce k` |
