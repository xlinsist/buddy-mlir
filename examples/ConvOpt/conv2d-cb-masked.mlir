// Set -conv-vectorization="strip-mining=4".

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 4)>
module {
  func.func @conv2d_cb_masked(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<4xf32>
    %1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %2 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %3 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %4 = memref.dim %arg2, %c1 : memref<?x?xf32>
    affine.for %arg3 = #map0(%c0) to #map0(%3) {
      affine.for %arg4 = #map0(%c0) to #map0(%1) {
        affine.for %arg5 = #map0(%c0) to #map0(%2) {
          affine.for %arg6 = #map0(%c0) to #map1(%4) {
            %5 = affine.vector_load %arg1[%arg4, %arg5] : memref<?x?xf32>, vector<1xf32>
            %6 = vector.broadcast %5 : vector<1xf32> to vector<4xf32>
            %7 = arith.muli %arg6, %c4 : index
            %8 = arith.subi %4, %7 : index
            %9 = arith.cmpi sge, %8, %c4 : index
            scf.if %9 {
              %10 = affine.vector_load %arg0[%arg3 + %arg4, %arg5 + %arg6 * 4] : memref<?x?xf32>, vector<4xf32>
              %11 = affine.vector_load %arg2[%arg3, %arg6 * 4] : memref<?x?xf32>, vector<4xf32>
              %12 = vector.fma %10, %6, %11 : vector<4xf32>
              affine.vector_store %12, %arg2[%arg3, %arg6 * 4] : memref<?x?xf32>, vector<4xf32>
            } else {
              %10 = vector.create_mask %8 : vector<4xi1>
              %11 = arith.addi %arg3, %arg4 : index
              %12 = arith.muli %arg6, %c4 : index
              %13 = arith.addi %arg5, %12 : index
              %14 = vector.maskedload %arg0[%11, %13], %10, %0 : memref<?x?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
              %15 = vector.maskedload %arg2[%arg3, %12], %10, %0 : memref<?x?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
              %16 = vector.fma %14, %6, %15 : vector<4xf32>
              vector.maskedstore %arg2[%arg3, %12], %10, %16 : memref<?x?xf32>, vector<4xi1>, vector<4xf32>
            }
          }
        }
      }
    }
    return
  }
}
