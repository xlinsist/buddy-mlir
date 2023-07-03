memref.global "private" @gv_i32 : memref<20x20xi32>

func.func @test() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20x20xi32>
  %result_mem = memref.get_global @gv_i32 : memref<20x20xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %mem_i32, %c0 : memref<20x20xi32>
  %dim_0 = memref.dim %mem_i32, %c1 : memref<20x20xi32>
  %dim_1 = memref.dim %mem_i32, %c1 : memref<20x20xi32>
  scf.for %idx0 = %c0 to %dim step %c1 {
    scf.for %idx1 = %c0 to %dim_1 step %c1 {
      scf.for %idx2 = %c0 to %dim_0 step %c1 {
        %0 = memref.load %mem_i32[%idx0, %idx2] : memref<20x20xi32>
        %1 = memref.load %mem_i32[%idx2, %idx1] : memref<20x20xi32>
        %2 = memref.load %result_mem[%idx0, %idx1] : memref<20x20xi32>
        %3 = arith.muli %0, %1 : i32
        %4 = arith.addi %2, %3 : i32
        memref.store %4, %result_mem[%idx0, %idx1] : memref<20x20xi32>
      }
    }
  }

  %result = vector.load %result_mem[%c0, %c0] : memref<20x20xi32>, vector<8xi32>

  %mask = arith.constant dense<1> : vector<8xi1>
  %c1_i32 = arith.constant 1 : i32
  %evl = arith.constant 8 : i32
  %res_reduce_add_mask_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %result, %mask, %evl) :
        (i32, vector<8xi32>, vector<8xi1>, i32) -> i32

  return %res_reduce_add_mask_driven : i32
}
