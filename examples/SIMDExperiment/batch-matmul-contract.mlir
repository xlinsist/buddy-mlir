func.func @contraction_batch_matmul(%arg0: memref<1x12x128xf32>, %arg1: memref<1x128x128xf32>, %arg2: memref<1x12x128xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x12x128xf32>, vector<1x12x128xf32>
    %1 = vector.broadcast %0 : vector<1x12x128xf32> to vector<128x1x12x128xf32>
    %2 = vector.transpose %1, [1, 2, 0, 3] : vector<128x1x12x128xf32> to vector<1x12x128x128xf32>
    %3 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x128x128xf32>, vector<1x128x128xf32>
    %4 = vector.broadcast %3 : vector<1x128x128xf32> to vector<12x1x128x128xf32>
    %5 = vector.transpose %4, [1, 0, 3, 2] : vector<12x1x128x128xf32> to vector<1x12x128x128xf32>
    %6 = vector.transfer_read %arg2[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x12x128xf32>, vector<1x12x128xf32>
    %7 = arith.mulf %2, %5 : vector<1x12x128x128xf32>
    %8 = vector.multi_reduction <add>, %7, %6 [3] : vector<1x12x128x128xf32> to vector<1x12x128xf32>
    vector.transfer_write %8, %arg2[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x12x128xf32>, memref<1x12x128xf32>
    return
  }
  