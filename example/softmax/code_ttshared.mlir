#map = affine_map<(d0) -> (d0)>
module {
  func.func @softmax_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.minsi %2, %c1024 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %4 = arith.cmpi slt, %3, %c1024 : index
    scf.if %4 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%3] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %6 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst_0 into %6[] : tensor<f32>
    %reduced = linalg.reduce ins(%5 : tensor<1024xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %16 = arith.maxnumf %in, %init : f32
        linalg.yield %16 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %7 = tensor.empty() : tensor<1024xf32>
    %8 = linalg.fill ins(%extracted : f32) outs(%7 : tensor<1024xf32>) -> tensor<1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %8 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %16 = arith.subf %in, %in_7 : f32
      linalg.yield %16 : f32
    } -> tensor<1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%9 : tensor<1024xf32>) outs(%9 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = math.exp %in : f32
      linalg.yield %16 : f32
    } -> tensor<1024xf32>
    %11 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst into %11[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%10 : tensor<1024xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %16 = arith.addf %in, %init : f32
        linalg.yield %16 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %12 = linalg.fill ins(%extracted_4 : f32) outs(%7 : tensor<1024xf32>) -> tensor<1024xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %12 : tensor<1024xf32>, tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %16 = arith.divf %in, %in_7 : f32
      linalg.yield %16 : f32
    } -> tensor<1024xf32>
    %14 = arith.muli %arg8, %arg3 : i32
    %15 = arith.index_cast %14 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%15], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %13[0] [%3] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%3] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

