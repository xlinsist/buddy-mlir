#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 4)>

memref.global "private" @gv_input_i32 : memref<128x128xi32>
memref.global "private" @gv_kernel_i32 : memref<3x3xi32>
memref.global "private" @gv_output_i32 : memref<126x126xi32>

func.func @test() -> i32 {

  %input1 = memref.get_global @gv_input_i32 : memref<128x128xi32>
  %input2 = memref.get_global @gv_kernel_i32 : memref<3x3xi32>
  %output = memref.get_global @gv_output_i32 : memref<126x126xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0 : i32
  %0 = vector.splat %cst : vector<4xi32>
  %1 = memref.dim %input2, %c0 : memref<3x3xi32>
  %2 = memref.dim %input2, %c1 : memref<3x3xi32>
  %3 = memref.dim %output, %c0 : memref<126x126xi32>
  %4 = memref.dim %output, %c1 : memref<126x126xi32>
  affine.for %idx0 = #map0(%c0) to #map0(%3) {
    affine.for %idx1 = #map0(%c0) to #map0(%1) {
      affine.for %idx2 = #map0(%c0) to #map0(%2) {
        affine.for %idx3 = #map0(%c0) to #map1(%4) {
          %5 = affine.vector_load %input2[%idx1, %idx2] : memref<126x126xi32>, vector<1xi32>
          %6 = vector.broadcast %5 : vector<1xi32> to vector<4xi32>
          %7 = arith.muli %idx3, %c4 : index
          %8 = arith.subi %4, %7 : index
          %9 = arith.cmpi sge, %8, %c4 : index
          scf.if %9 {
            %10 = affine.vector_load %input1[%idx0 + %idx1, %idx2 + %idx3 * 4] : memref<128x128xi32>, vector<4xi32>
            %11 = affine.vector_load %output[%idx0, %idx3 * 4] : memref<126x126xi32>, vector<4xi32>
            %12 = arith.muli %10, %6 : vector<4xi32>
            %13 = arith.addi %12, %11 : vector<4xi32>
            affine.vector_store %13, %output[%idx0, %idx3 * 4] : memref<126x126xi32>, vector<4xi32>
          } else {
            %10 = arith.constant dense<1> : vector<4xi1>
            %11 = arith.addi %idx0, %idx1 : index
            %12 = arith.muli %idx3, %c4 : index
            %13 = arith.addi %idx2, %12 : index
            %14 = vector.maskedload %input1[%11, %13], %10, %0 : memref<128x128xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
            %15 = vector.maskedload %output[%idx0, %12], %10, %0 : memref<126x126xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
            %16 = arith.muli %14, %6 : vector<4xi32>
            %17 = arith.addi %16, %15 : vector<4xi32>
            vector.maskedstore %output[%idx0, %12], %10, %17 : memref<126x126xi32>, vector<4xi1>, vector<4xi32>
          }
        }
      }
    }
  }

  %result = vector.load %output[%c0, %c0] : memref<126x126xi32>, vector<8xi32>

  %mask = arith.constant dense<1> : vector<8xi1>
  %c1_i32 = arith.constant 1 : i32
  %evl = arith.constant 8 : i32
  %res_reduce_add_mask_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %result, %mask, %evl) :
        (i32, vector<8xi32>, vector<8xi1>, i32) -> i32

  return %res_reduce_add_mask_driven : i32
}
