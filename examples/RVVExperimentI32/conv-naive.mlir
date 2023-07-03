#map = affine_map<(d0, d1) -> (d0 + d1)>

memref.global "private" @gv_input_i32 : memref<128x128xi32>
memref.global "private" @gv_kernel_i32 : memref<3x3xi32>
memref.global "private" @gv_output_i32 : memref<126x126xi32>

func.func @test() -> i32 {

  %input1 = memref.get_global @gv_input_i32 : memref<128x128xi32>
  %input2 = memref.get_global @gv_kernel_i32 : memref<3x3xi32>
  %output = memref.get_global @gv_output_i32 : memref<126x126xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %input2, %c0 : memref<3x3xi32>
  %dim_0 = memref.dim %input2, %c1 : memref<3x3xi32>
  %dim_1 = memref.dim %output, %c0 : memref<126x126xi32>
  %dim_2 = memref.dim %output, %c1 : memref<126x126xi32>
  scf.for %idx0 = %c0 to %dim_1 step %c1 {
    scf.for %idx1 = %c0 to %dim_2 step %c1 {
      scf.for %idx2 = %c0 to %dim step %c1 {
        scf.for %idx3 = %c0 to %dim_0 step %c1 {
          %0 = affine.apply #map(%idx0, %idx2)
          %1 = affine.apply #map(%idx1, %idx3)
          %2 = memref.load %input1[%0, %1] : memref<128x128xi32>
          %3 = memref.load %input2[%idx2, %idx3] : memref<3x3xi32>
          %4 = memref.load %output[%idx0, %idx1] : memref<126x126xi32>
          %5 = arith.muli %2, %3 : i32
          %6 = arith.addi %4, %5 : i32
          memref.store %6, %output[%idx0, %idx1] : memref<126x126xi32>
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
