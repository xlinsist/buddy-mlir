memref.global "private" @gv_i32 : memref<32x32xi32>

func.func @test() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<32x32xi32>
  %result_mem = memref.get_global @gv_i32 : memref<32x32xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32

  %aRow = memref.dim %mem_i32, %c0 : memref<32x32xi32>
  %aRowMinus1 = arith.subi %aRow, %c1 : index
  %aCol = memref.dim %mem_i32, %c1 : memref<32x32xi32>
  %bRow = memref.dim %mem_i32, %c0 : memref<32x32xi32>
  %bCol = memref.dim %mem_i32, %c1 : memref<32x32xi32>
  %bCol_i32 = arith.index_cast %bCol : index to i32

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : i32
  // LMUL = 4
  %lmul = arith.constant 2 : i32

  affine.for %idx0 = 0 to %aCol {
    affine.for %idx1 = 0 to %aRow step 7  {
      
      %aEleRowAdd0 = arith.addi %idx1, %c0 : index
      %aEleRow0 = arith.minui %aRowMinus1, %aEleRowAdd0 : index
      %aEleRowAdd1 = arith.addi %idx1, %c1 : index
      %aEleRow1 = arith.minui %aRowMinus1, %aEleRowAdd1 : index
      %aEleRowAdd2 = arith.addi %idx1, %c2 : index
      %aEleRow2 = arith.minui %aRowMinus1, %aEleRowAdd2 : index
      %aEleRowAdd3 = arith.addi %idx1, %c3 : index
      %aEleRow3 = arith.minui %aRowMinus1, %aEleRowAdd3 : index
      %aEleRowAdd4 = arith.addi %idx1, %c4 : index
      %aEleRow4 = arith.minui %aRowMinus1, %aEleRowAdd4 : index
      %aEleRowAdd5 = arith.addi %idx1, %c5 : index
      %aEleRow5 = arith.minui %aRowMinus1, %aEleRowAdd5 : index
      %aEleRowAdd6 = arith.addi %idx1, %c6 : index
      %aEleRow6 = arith.minui %aRowMinus1, %aEleRowAdd6 : index

      %aEleVector0 = vector.load %mem_i32[%aEleRow0, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector1 = vector.load %mem_i32[%aEleRow1, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector2 = vector.load %mem_i32[%aEleRow2, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector3 = vector.load %mem_i32[%aEleRow3, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector4 = vector.load %mem_i32[%aEleRow4, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector5 = vector.load %mem_i32[%aEleRow5, %idx0] : memref<32x32xi32>, vector<[8]xi32>
      %aEleVector6 = vector.load %mem_i32[%aEleRow6, %idx0] : memref<32x32xi32>, vector<[8]xi32>

      // While loop for strip-mining.
      %tmpAVL, %tmpIdx = scf.while (%avl = %bCol_i32, %idx = %c0) : (i32, index) -> (i32, index) {
        // If avl greater than zero.
        %cond = arith.cmpi sgt, %avl, %c0_i32 : i32
        // Pass avl, idx to the after region.
        scf.condition(%cond) %avl, %idx : i32, index
      } do {
      ^bb0(%avl : i32, %idx : index):
        // Perform the calculation according to the vl.
        %vl = rvv.setvl %avl, %sew, %lmul : i32
        %mask = arith.constant dense<1> : vector<[8]xi1>
        %input_vector = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %mem_i32[%idx0, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %mul_vector0 = rvv.mul %input_vector, %aEleVector0, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector1 = rvv.mul %input_vector, %aEleVector1, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector2 = rvv.mul %input_vector, %aEleVector2, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector3 = rvv.mul %input_vector, %aEleVector3, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector4 = rvv.mul %input_vector, %aEleVector4, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector5 = rvv.mul %input_vector, %aEleVector5, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %mul_vector6 = rvv.mul %input_vector, %aEleVector6, %vl : vector<[8]xi32>, vector<[8]xi32>, i32

        %c_vector0 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow0, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector1 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow1, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector2 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow2, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector3 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow3, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector4 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow4, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector5 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow5, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>
        %c_vector6 = vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          %ele = vector.load %result_mem[%aEleRow6, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield %ele : vector<[8]xi32>
        } : vector<[8]xi32>

        %result_vector0 = rvv.add %mul_vector0, %c_vector0, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector1 = rvv.add %mul_vector1, %c_vector1, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector2 = rvv.add %mul_vector2, %c_vector2, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector3 = rvv.add %mul_vector3, %c_vector3, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector4 = rvv.add %mul_vector4, %c_vector4, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector5 = rvv.add %mul_vector5, %c_vector5, %vl : vector<[8]xi32>, vector<[8]xi32>, i32
        %result_vector6 = rvv.add %mul_vector6, %c_vector6, %vl : vector<[8]xi32>, vector<[8]xi32>, i32

        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector0, %result_mem[%aEleRow0, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector1, %result_mem[%aEleRow1, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector2, %result_mem[%aEleRow2, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector3, %result_mem[%aEleRow3, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector4, %result_mem[%aEleRow4, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector5, %result_mem[%aEleRow5, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()
        vector_exp.predication %mask, %vl : vector<[8]xi1>, i32 {
          vector.store %result_vector6, %result_mem[%aEleRow6, %idx] : memref<32x32xi32>, vector<[8]xi32>
          vector.yield
        } : () -> ()

        // Update idx and avl.
        %vl_ind = arith.index_cast %vl : i32 to index
        %new_idx = arith.addi %idx, %vl_ind : index
        %new_avl = arith.subi %avl, %vl : i32
        scf.yield %new_avl, %new_idx : i32, index
      }
    }
  }

  %result = vector.load %result_mem[%c0, %c0] : memref<32x32xi32>, vector<8xi32>

  %mask = arith.constant dense<1> : vector<8xi1>
  %c1_i32 = arith.constant 1 : i32
  %evl = arith.constant 8 : i32
  %res_reduce_add_mask_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %result, %mask, %evl) :
        (i32, vector<8xi32>, vector<8xi1>, i32) -> i32

  return %res_reduce_add_mask_driven : i32
}
