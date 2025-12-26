# Triton GPU 到 TVM Dialect 转换映射详细文档

## 概述

本文档详细描述了 Triton GPU IR (ttgir) 中各个 dialect 操作如何转换为 TVM dialect 操作。转换过程通过多个 pass 实现，主要包括：

1. **LowerToTensorIdioms**: 将 Triton 操作转换为标准 Tensor dialect
2. **MaterializeTensorsToTVMBuffers**: 将 Tensor 操作物化为 TVM buffer 操作
3. **ReplaceTritonPointersWithMemRefs**: 将 Triton 指针替换为 MemRef
4. **ConvertToTVMScript**: 将 TVM dialect 转换为 Python TensorIR 脚本

## 转换管道架构

### 类型转换流程

类型转换贯穿整个转换管道，具体流程如下：

```
Triton Types → MLIR Standard Types → TVM Dialect Types → TVM Script Types
     |               |                      |                  |
  !tt.ptr<T>    →  memref<?xT>        →  tvm.ref        →  T.handle
tensor<NxT>     →  tensor<NxT>        →  memref<NxT>     →  T.Buffer
     T          →       T              →       T         →   "dtype"
```

**转换阶段说明**：
1. **Pass1 (LowerToTensorIdioms)**: 保持标准MLIR类型，主要转换操作语义
2. **Pass2 (ReplaceTritonPointersWithMemRefs)**: 将Triton指针类型转换为MemRef
3. **Pass3 (MaterializeTensorsToTVMBuffers)**: 将Tensor转换为TVM Buffer操作
4. **Pass4 (ConvertToTVMScript)**: 将类型转换为TVM Script字符串表示

### 转换管道详细流程

```
Triton GPU IR → Tensor IR → TVM Dialect → TVM Script
    |             |           |            |
   Pass1        Pass2       Pass3       Pass4
```

## 详细转换映射

### 1. Triton Core Dialect 转换

#### 1.1 `tt.splat` 操作
**转换前 (Triton)**:
```mlir
%result = tt.splat %src : (f32) -> tensor<128xf32>
```

**转换后 (Tensor)**:
```mlir
%result = tensor.generate {
  ^bb0(%arg0: index):
    tensor.yield %src : f32
}
```

**最终 TVM**:
```mlir
tvm.block {
  %axis = tvm.axis spatial %extent = %iv : index
  %ref = tvm.ref %buffer[%axis] : memref<128xf32>
  tvm.write %ref : f32
  tvm.assign %ref = %src : f32
}
```

#### 1.2 `tt.make_range` 操作
**转换前 (Triton)**:
```mlir
%result = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
```

**转换后 (Tensor)**:
```mlir
%result = tensor.generate {
  ^bb0(%arg0: index):
    %cast = arith.index_cast %arg0 : index to i32
    tensor.yield %cast : i32
}
```

**最终 TVM**:
```mlir
tvm.block {
  %axis = tvm.axis spatial %extent = %iv : index
  %ref = tvm.ref %buffer[%axis] : memref<128xi32>
  tvm.write %ref : i32
  tvm.assign %ref = %axis : i32
}
```

#### 1.3 `tt.addptr` 操作
**转换前 (Triton)**:
```mlir
%result = tt.addptr %ptr, %offset : !tt.ptr<f32>, tensor<128xi32>
```

**转换后 (Tensor)**:
```mlir
%result = tensor.generate {
  ^bb0(%arg0: index):
    %scalar_ptr = tensor.extract %ptr[%arg0] : tensor<128x!tt.ptr<f32>>
    %scalar_offset = tensor.extract %offset[%arg0] : tensor<128xi32>
    %new_ptr = tt.addptr %scalar_ptr, %scalar_offset : !tt.ptr<f32>, i32
    tensor.yield %new_ptr : !tt.ptr<f32>
}
```

#### 1.4 `tt.load` 操作
**转换前 (Triton)**:
```mlir
%result = tt.load %ptr, %mask, %other : tensor<128x!tt.ptr<f32>>
```

**转换后 (TVM)**:
```mlir
%buffer = tvm.alloc_buffer() {scope = "shared"} : memref<128xf32>
scf.for %iv = %c0 to %c128 step %c1 {tvm.for_kind = serial} {
  tvm.block {
    %axis = tvm.axis spatial %extent = %iv : index
    
    // 从指针生成器中获取内存引用
    %ptr_scalar = <inlined from ptr generator>
    %memref_ptr = // 从 ttm.memref_to_ptr 获取
    %src_ref = tvm.ref %memref[%indices] : memref<...xf32>
    
    tvm.read %src_ref : f32
    
    // 如果有掩码，添加条件
    %mask_val = <inlined from mask generator>
    %other_val = <inlined from other generator>
    %value = tvm.if_then_else %mask_val, %src_ref, %other_val : f32
    
    %dest_ref = tvm.ref %buffer[%axis] : memref<128xf32>
    tvm.write %dest_ref : f32
    tvm.assign %dest_ref = %value : f32
  }
}
%result = ttm.memref_to_tensor %buffer : memref<128xf32> to tensor<128xf32>
```

#### 1.5 `tt.store` 操作
**转换前 (Triton)**:
```mlir
tt.store %ptr, %value, %mask : tensor<128x!tt.ptr<f32>>
```

**转换后 (TVM)**:
```mlir
scf.for %iv = %c0 to %c128 step %c1 {tvm.for_kind = serial} {
  tvm.block {
    %axis = tvm.axis spatial %extent = %iv : index
    
    // 如果有掩码，添加条件
    %mask_val = <inlined from mask generator>
    tvm.where %mask_val
    
    // 从指针生成器中获取目标引用
    %ptr_scalar = <inlined from ptr generator>
    %memref_ptr = // 从 ttm.memref_to_ptr 获取
    %dest_ref = tvm.ref %memref[%indices] : memref<...xf32>
    
    // 提取要存储的值
    %store_value = tensor.extract %value[%axis] : tensor<128xf32>
    
    tvm.write %dest_ref : f32
    tvm.assign %dest_ref = %store_value : f32
  }
}
```

#### 1.6 `tt.reduce` 操作
**转换前 (Triton)**:
```mlir
%result = tt.reduce %input {
  ^bb0(%arg0: f32, %arg1: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %sum : f32
} : tensor<128x256xf32> -> tensor<128xf32>
```

**转换后 (TVM)**:
```mlir
%buffer = tvm.alloc_buffer() {scope = "shared"} : memref<128xf32>
scf.for %i = %c0 to %c128 step %c1 {tvm.for_kind = serial} {
  scf.for %j = %c0 to %c256 step %c1 {tvm.for_kind = serial} {
    tvm.block {
      %i_axis = tvm.axis spatial %c128 = %i : index
      %j_axis = tvm.axis reduce %c256 = %j : index
      
      %acc_ref = tvm.ref %buffer[%i_axis] : memref<128xf32>
      tvm.write %acc_ref : f32
      
      // 初始化块
      tvm.init {
        %zero = arith.constant 0.0 : f32
        tvm.assign %acc_ref = %zero : f32
      }
      
      // 累加操作
      %input_val = tensor.extract %input[%i_axis, %j_axis] : tensor<128x256xf32>
      %current_acc = %acc_ref // 当前累加值的引用
      %new_acc = arith.addf %current_acc, %input_val : f32
      tvm.assign %acc_ref = %new_acc : f32
    }
  }
}
%result = ttm.memref_to_tensor %buffer : memref<128xf32> to tensor<128xf32>
```

### 2. Arithmetic 和 Math Dialect 转换

#### 2.1 基本算术操作
算术操作通常保持不变，直接传递到最终的 TVM 脚本：

| Triton/MLIR 操作 | TVM Dialect | TVM Script |
|-----------------|-------------|------------|
| `arith.addf` | `arith.addf` | `+` |
| `arith.subf` | `arith.subf` | `-` |
| `arith.mulf` | `arith.mulf` | `*` |
| `arith.divf` | `arith.divf` | `/` |
| `arith.addi` | `arith.addi` | `+` |
| `arith.subi` | `arith.subi` | `-` |
| `arith.muli` | `arith.muli` | `*` |

#### 2.2 比较操作
| Triton/MLIR 操作 | TVM Dialect | TVM Script |
|-----------------|-------------|------------|
| `arith.cmpf(eq)` | `arith.cmpf(eq)` | `==` |
| `arith.cmpf(ne)` | `arith.cmpf(ne)` | `!=` |
| `arith.cmpf(lt)` | `arith.cmpf(lt)` | `<` |
| `arith.cmpf(le)` | `arith.cmpf(le)` | `<=` |
| `arith.cmpf(gt)` | `arith.cmpf(gt)` | `>` |
| `arith.cmpf(ge)` | `arith.cmpf(ge)` | `>=` |

#### 2.3 数学函数
| Triton/MLIR 操作 | TVM Dialect | TVM Script |
|-----------------|-------------|------------|
| `math.exp` | `math.exp` | `T.exp()` |
| `math.log` | `math.log` | `T.log()` |
| `math.sqrt` | `math.sqrt` | `T.sqrt()` |
| `math.sin` | `math.sin` | `T.sin()` |
| `math.cos` | `math.cos` | `T.cos()` |
| `math.floor` | `math.floor` | `T.floor()` |
| `math.ceil` | `math.ceil` | `T.ceil()` |

### 3. 类型转换对应关系

#### 3.1 基础数据类型映射

| Triton/MLIR 类型 | TVM Dialect 类型 | TVM Script 类型 | 说明 |
|------------------|------------------|-----------------|------|
| `f32` | `f32` | `"float32"` | 32位浮点数 |
| `f16` | `f16` | `"float16"` | 16位半精度浮点数 |
| `bf16` | `bf16` | `"bfloat16"` | 16位bfloat格式 |
| `f64` | `f64` | `"float64"` | 64位双精度浮点数 |
| `i1` | `i1` | `"bool"` | 布尔类型 |
| `i8` | `i8` | `"int8"` | 8位有符号整数 |
| `ui8` | `ui8` | `"uint8"` | 8位无符号整数 |
| `i16` | `i16` | `"int16"` | 16位有符号整数 |
| `ui16` | `ui16` | `"uint16"` | 16位无符号整数 |
| `i32` | `i32` | `"int32"` | 32位有符号整数 |
| `ui32` | `ui32` | `"uint32"` | 32位无符号整数 |
| `i64` | `i64` | `"int64"` | 64位有符号整数 |
| `ui64` | `ui64` | `"uint64"` | 64位无符号整数 |
| `index` | `index` | `"int32"` | 索引类型（通常转换为i32） |

#### 3.2 张量类型映射

**Triton Tensor Types → TVM Buffer Types**:
```mlir
// Triton
tensor<128x256xf32, #blocked_layout>

// TVM Dialect (中间表示)
memref<128x256xf32>

// TVM Script (最终表示)
T.Buffer((128, 256), "float32")
```

**编码信息处理**:
- Triton的`BlockedEncodingAttr`被转换为TVM的循环嵌套结构
- `warpsPerCTA`、`threadsPerWarp`、`sizePerThread`影响并行化策略

#### 3.3 指针类型映射

| Triton 指针类型 | 中间表示 | TVM Script 表示 |
|-----------------|----------|----------------|
| `!tt.ptr<f32>` | `memref<?xf32>` | `T.handle` + `T.match_buffer` |
| `tensor<128x!tt.ptr<f32>>` | `memref<128x?xf32>` | 多维Buffer访问 |

**转换示例**:
```mlir
// Triton
func.func @example(%ptr: !tt.ptr<f32>) {
  %data = tt.load %ptr : !tt.ptr<f32>
}

// TVM Dialect
func.func @example(%memref: memref<?xf32>) {
  %ref = tvm.ref %memref[%indices] : memref<?xf32>
  tvm.read %ref : f32
}

// TVM Script
@T.prim_func
def example(A: T.handle):
    A_buffer = T.match_buffer(A, (), dtype="float32")
    # 访问和读取逻辑
```

#### 3.4 类型转换操作

| MLIR 转换操作 | TVM Dialect | TVM Script |
|---------------|-------------|------------|
| `arith.index_cast` | `arith.index_cast` | 自动处理 |
| `arith.sitofp` | `arith.sitofp` | `T.cast()` |
| `arith.fptosi` | `arith.fptosi` | `T.cast()` |
| `arith.fptrunc` | `arith.fptrunc` | `T.cast()` |
| `arith.fpext` | `arith.fpext` | `T.cast()` |
| `arith.trunci` | `arith.trunci` | `T.cast()` |
| `arith.extsi` | `arith.extsi` | `T.cast()` |
| `arith.extui` | `arith.extui` | `T.cast()` |

#### 3.5 常量类型处理

```mlir
// Triton 常量
%c0_f32 = arith.constant 0.0 : f32
%c1_i32 = arith.constant 1 : i32

// TVM Dialect (保持不变)
%c0_f32 = arith.constant 0.0 : f32
%c1_i32 = arith.constant 1 : i32

// TVM Script (内联到表达式中)
// 0.0 和 1 直接作为字面量使用
```

#### 3.6 特殊类型处理

**Splat 常量张量**:
```mlir
// Triton
%splat = tt.splat %scalar : (f32) -> tensor<128xf32>

// TVM Dialect
%buffer = tvm.alloc_buffer() : memref<128xf32>
scf.for %i = %c0 to %c128 step %c1 {tvm.for_kind = unroll} {
  tvm.block {
    %axis = tvm.axis spatial %c128 = %i : index
    %ref = tvm.ref %buffer[%axis] : memref<128xf32>
    tvm.write %ref : f32
    tvm.assign %ref = %scalar : f32
  }
}
```

**Range 张量**:
```mlir
// Triton
%range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>

// TVM Dialect
%buffer = tvm.alloc_buffer() : memref<128xi32>
scf.for %i = %c0 to %c128 step %c1 {tvm.for_kind = unroll} {
  tvm.block {
    %axis = tvm.axis spatial %c128 = %i : index
    %ref = tvm.ref %buffer[%axis] : memref<128xi32>
    tvm.write %ref : i32
    %cast = arith.index_cast %axis : index to i32
    tvm.assign %ref = %cast : i32
  }
}
```

#### 3.7 类型兼容性限制

**当前不支持的类型**:
- 复数类型 (`complex<f32>`, `complex<f64>`)
- 动态形状张量（除非在编译时已知）
- 嵌套结构类型
- 函数类型

**类型转换限制**:
- 所有张量维度必须在编译时确定
- 指针类型只能作为函数参数传入
- 索引类型统一转换为`i32`以兼容Triton的限制

### 4. TritonGPU Dialect 转换

#### 4.1 Layout 和 Encoding
TritonGPU 的 layout 和 encoding 信息在转换到 TVM 时主要用于：
- 确定循环嵌套结构
- 决定线程绑定策略
- 影响内存访问模式

#### 4.2 GPU 特定操作转换

**Block 和 Thread 维度映射**:
```mlir
// Triton GPU 隐式线程模型
scf.for %block_x = ... {
  scf.for %thread_x = ... {tvm.for_kind = thread_binding, tvm.for_thread = "threadIdx.x"}
    // 计算逻辑
  }
}
```

### 5. 控制流转换

#### 5.1 循环结构
**标准循环**:
```mlir
// Triton
scf.for %i = %lb to %ub step %step {
  // body
}

// TVM
scf.for %i = %lb to %ub step %step {tvm.for_kind = serial} {
  // body
}
```

**并行循环**:
```mlir
scf.parallel (%i) = (%lb) to (%ub) step (%step) {
  // body
}

// 转换为
scf.for %i = %lb to %ub step %step {tvm.for_kind = parallel} {
  // body
}
```

### 6. 内存管理

#### 6.1 Buffer 分配
```mlir
// TVM Dialect
%buffer = tvm.alloc_buffer() {scope = "global"} : memref<128x256xf32>
%buffer_shared = tvm.alloc_buffer() {scope = "shared"} : memref<32x32xf32>
%buffer_local = tvm.alloc_buffer() {scope = "local"} : memref<16xf32>
```

对应的 TVM Script:
```python
A = T.alloc_buffer((128, 256), "float32")
A_shared = T.alloc_buffer((32, 32), "float32", scope="shared")
A_local = T.alloc_buffer((16,), "float32", scope="local")
```

#### 6.2 Buffer 访问
```mlir
// TVM Dialect
%ref = tvm.ref %buffer[%i, %j] : memref<128x256xf32>
tvm.read %ref : f32
tvm.write %ref : f32
tvm.assign %ref = %value : f32
```

对应的 TVM Script:
```python
T.reads(A[i, j])
T.writes(A[i, j])
A[i, j] = value
```

### 7. 轴和迭代器转换

#### 7.1 轴绑定
```mlir
// TVM Dialect
%spatial_axis = tvm.axis spatial %extent = %iv : index
%reduce_axis = tvm.axis reduce %extent = %iv : index
```

对应的 TVM Script:
```python
i = T.axis.spatial(extent)
k = T.axis.reduce(extent)
```

### 8. 特殊操作

#### 8.1 条件表达式
```mlir
// TVM Dialect
%result = tvm.if_then_else %cond, %true_val, %false_val : f32
```

对应的 TVM Script:
```python
T.if_then_else(cond, true_val, false_val)
```

#### 8.2 块组织
```mlir
// TVM Dialect
tvm.block {
  %axis = tvm.axis spatial %extent = %iv : index
  tvm.read %input_ref : f32
  tvm.write %output_ref : f32
  tvm.assign %output_ref = %computed_value : f32
}
```

对应的 TVM Script:
```python
with T.block("block_name"):
    i = T.axis.spatial(extent)
    T.reads(input[i])
    T.writes(output[i])
    output[i] = computed_value
```

## 转换限制

当前实现有以下限制：

1. **不支持的操作**:
   - `tt.make_tensor_ptr` 和 `tt.advance`
   - `tt.dot` (矩阵乘法)
   - 动态形状 (必须标记为 `tl.constexpr`)

2. **类型限制**:
   - 只支持指针类型的函数参数
   - 整数参数必须是常量

3. **内存限制**:
   - 主要针对单GPU执行
   - 有限的内存层次支持

## 优化策略

转换过程中应用的主要优化：

1. **循环融合**: 相同类型的相邻循环会被融合
2. **常量传播**: 编译时常量会被内联
3. **死代码消除**: 移除未使用的张量生成
4. **循环不变代码外提**: 将循环不变计算移出循环

## 示例：完整转换流程

### 输入 Triton 代码
```python
@triton.jit
def vector_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 中间IR：类型转换过程

**阶段1：Triton GPU IR**
```mlir
func.func @vector_add(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, 
                      %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
  // 指针类型: !tt.ptr<f32>
  // 张量类型: tensor<128xf32, #blocked_layout>
}
```

**阶段2：指针转换为MemRef**
```mlir
func.func @vector_add(%x: memref<?xf32>, %y: memref<?xf32>, 
                      %output: memref<?xf32>, %n_elements: i32) {
  // 指针类型转换: !tt.ptr<f32> → memref<?xf32>
  // 张量类型保持: tensor<128xf32>
}
```

**阶段3：张量物化为TVM Buffer**
```mlir
func.func @vector_add(%x: memref<?xf32>, %y: memref<?xf32>, 
                      %output: memref<?xf32>, %n_elements: i32) {
  // 分配中间Buffer
  %buffer_x = tvm.alloc_buffer() : memref<128xf32>
  %buffer_y = tvm.alloc_buffer() : memref<128xf32>
  %buffer_out = tvm.alloc_buffer() : memref<128xf32>
  
  // 类型已统一为memref<NxT>
  scf.for %i = %c0 to %c128 step %c1 {tvm.for_kind = "thread_binding"} {
    tvm.block {
      %axis = tvm.axis spatial %c128 = %i : index
      // f32类型在计算中保持不变
    }
  }
}
```

### 最终 TVM Script
```python
@T.prim_func
def vector_add(x: T.handle, y: T.handle, output: T.handle, n_elements: T.int32):
    # 类型最终表示: T.handle → T.match_buffer → "float32"
    X = T.match_buffer(x, (n_elements,), dtype="float32")
    Y = T.match_buffer(y, (n_elements,), dtype="float32")
    Output = T.match_buffer(output, (n_elements,), dtype="float32")
    
    for i in T.thread_binding(0, T.ceildiv(n_elements, BLOCK_SIZE), thread="blockIdx.x"):
        for j in T.thread_binding(0, BLOCK_SIZE, thread="threadIdx.x"):
            with T.block("compute"):
                idx = T.axis.spatial(n_elements, i * BLOCK_SIZE + j)
                T.where(idx < n_elements)
                T.reads(X[idx], Y[idx])
                T.writes(Output[idx])
                # 最终计算使用原生float32运算
                Output[idx] = X[idx] + Y[idx]
```

### 类型转换总结

**完整的类型转换映射**：
1. **函数参数**: `!tt.ptr<f32>` → `memref<?xf32>` → `T.handle` + `T.match_buffer(..., dtype="float32")`
2. **中间张量**: `tensor<128xf32>` → `memref<128xf32>` → `T.alloc_buffer(..., dtype="float32")`
3. **元素类型**: `f32` → `f32` → `"float32"`
4. **标量常量**: `arith.constant 0.0 : f32` → `arith.constant 0.0 : f32` → `0.0`
5. **索引类型**: `index` → `i32` → `T.int32` (为了兼容Triton)

这个转换文档提供了从 Triton GPU dialect 到 TVM dialect 的详细映射关系，涵盖了所有主要的操作类型和转换模式。
