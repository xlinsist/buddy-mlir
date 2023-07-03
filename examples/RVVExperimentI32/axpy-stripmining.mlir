memref.global "private" @gv_i32 : memref<262145xi32> // 262145 = 256 * 1024 + 1

func.func @test() -> i32 {

  %input1 = memref.get_global @gv_i32 : memref<262145xi32>
  %input2 = memref.get_global @gv_i32 : memref<262145xi32>
  %output = memref.get_global @gv_i32 : memref<262145xi32>

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %input1, %c0 : memref<262145xi32>
  %dim_i32 = arith.index_cast %dim : index to i32

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : i32
  // LMUL = 8
  %lmul = arith.constant 3 : i32

  // Constant mask configuration.
  %mask = arith.constant dense<1> : vector<[16]xi1>
  %a_element = affine.load %input1[%c0] : memref<262145xi32>

  // While loop for strip-mining.
  %tmp_avl, %tmp_idx = scf.while (%avl = %dim_i32, %idx = %c0) : (i32, index) -> (i32, index) {
    // If avl greater than zero.
    %cond = arith.cmpi sgt, %avl, %c0_i32 : i32
    // Pass avl, idx to the after region.
    scf.condition(%cond) %avl, %idx : i32, index
  } do {
  ^bb0(%avl : i32, %idx : index):
    // Perform the calculation according to the vl.
    %vl = rvv.setvl %avl, %sew, %lmul : i32
    %x_vector = vector_exp.predication %mask, %vl : vector<[16]xi1>, i32 {
      %ele = vector.load %input1[%idx] : memref<262145xi32>, vector<[16]xi32>
      vector.yield %ele : vector<[16]xi32>
    } : vector<[16]xi32>
    %y_vector = vector_exp.predication %mask, %vl : vector<[16]xi1>, i32 {
      %ele = vector.load %input2[%idx] : memref<262145xi32>, vector<[16]xi32>
      vector.yield %ele : vector<[16]xi32>
    } : vector<[16]xi32>
    %mul_vector = rvv.mul %x_vector, %a_element, %vl : vector<[16]xi32>, i32, i32
    %result_vector = rvv.add %mul_vector, %y_vector, %vl : vector<[16]xi32>, vector<[16]xi32>, i32
    vector_exp.predication %mask, %vl : vector<[16]xi1>, i32 {
      vector.store %result_vector, %output[%idx] : memref<262145xi32>, vector<[16]xi32>
      vector.yield
    } : () -> ()
    // Update idx and avl.
    %vl_ind = arith.index_cast %vl : i32 to index
    %new_idx = arith.addi %idx, %vl_ind : index
    %new_avl = arith.subi %avl, %vl : i32
    scf.yield %new_avl, %new_idx : i32, index
  }

  %result = vector.load %output[%c0] : memref<262145xi32>, vector<8xi32>

  %mask_res = arith.constant dense<1> : vector<8xi1>
  %c1_i32 = arith.constant 1 : i32
  %evl = arith.constant 8 : i32
  %res_reduce_add_mask_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %result, %mask_res, %evl) :
        (i32, vector<8xi32>, vector<8xi1>, i32) -> i32

  return %res_reduce_add_mask_driven : i32
}
