#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = arith.addi %arg3, %c31_i32 : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.addi %arg4, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg12, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg12, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %14, %c64_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %arg3 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.muli %16, %20 : index
    %22 = arith.muli %19, %20 : index
    %23 = arith.index_cast %arg7 : i32 to index
    %24 = arith.index_cast %arg4 : i32 to index
    %25 = arith.addi %arg5, %c15_i32 : i32
    %26 = arith.divsi %25, %c16_i32 : i32
    %27 = arith.muli %arg7, %c16_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29:3 = scf.for %arg15 = %c0_i32 to %26 step %c1_i32 iter_args(%arg16 = %1, %arg17 = %21, %arg18 = %c0) -> (tensor<32x64xf32>, index, index)  : i32 {
      %41 = arith.addi %arg18, %18 : index
      %42 = arith.remsi %41, %24 : index
      %43 = arith.subi %41, %42 : index
      %44 = arith.addi %42, %c64 : index
      %45 = arith.minsi %44, %24 : index
      %46 = arith.subi %45, %42 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [%c16, %46], strides: [%23, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
      %47 = arith.subi %c64, %46 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%c16, %47], strides: [%23, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
      %48 = arith.remsi %arg17, %20 : index
      %49 = arith.addi %22, %48 : index
      %50 = arith.subi %49, %arg17 : index
      %51 = arith.divsi %50, %20 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg17], sizes: [%51, %c16], strides: [%20, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
      %52 = arith.subi %c32, %51 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%48], sizes: [%52, %c16], strides: [%20, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
      %53 = arith.muli %arg15, %c16_i32 : i32
      %54 = arith.subi %arg5, %53 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.minsi %55, %c16 : index
      %alloc = memref.alloc() : memref<32x16xf32>
      %57 = arith.cmpi slt, %56, %c16 : index
      scf.if %57 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<32x16xf32>)
      }
      %58 = arith.minsi %51, %c32 : index
      %59 = arith.subi %c32, %58 : index
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%58, %56] [1, 1] : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%59, %56] [1, 1] : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%58, %56] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_7 = memref.subview %alloc[%58, 0] [%59, %56] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      memref.copy %subview_4, %subview_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
      memref.copy %subview_5, %subview_7 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      %60 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32>
      %alloc_8 = memref.alloc() : memref<16x64xf32>
      scf.if %57 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<16x64xf32>)
      }
      %61 = arith.minsi %46, %c64 : index
      %62 = arith.subi %c64, %61 : index
      %subview_9 = memref.subview %reinterpret_cast_0[0, 0] [%56, %61] [1, 1] : memref<16x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %reinterpret_cast_1[0, 0] [%56, %62] [1, 1] : memref<16x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %alloc_8[0, 0] [%56, %61] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      %subview_12 = memref.subview %alloc_8[0, %61] [%56, %62] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_9, %subview_11 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_10, %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %63 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x64xf32>
      %64 = linalg.matmul ins(%60, %63 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %65 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%64, %arg16 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%64 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %68 = arith.addf %in, %in_13 : f32
        linalg.yield %68 : f32
      } -> tensor<32x64xf32>
      %66 = arith.addi %arg17, %c16 : index
      %67 = arith.addi %arg18, %28 : index
      scf.yield %65, %66, %67 : tensor<32x64xf32>, index, index
    }
    %30 = arith.index_cast %arg8 : i32 to index
    %31 = arith.muli %16, %30 : index
    %32 = arith.addi %31, %18 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [32, 64], strides: [%30, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
    %33 = arith.addi %16, %c32 : index
    %34 = arith.minsi %33, %19 : index
    %35 = arith.subi %34, %16 : index
    %36 = arith.addi %18, %c64 : index
    %37 = arith.minsi %36, %24 : index
    %38 = arith.subi %37, %18 : index
    %39 = arith.minsi %35, %c32 : index
    %40 = arith.minsi %38, %c64 : index
    %extracted_slice = tensor.extract_slice %29#0[0, 0] [%39, %40] [1, 1] : tensor<32x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%39, %40] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

