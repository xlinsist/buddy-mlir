module {
  func.func @conv2d_cb_stripmining(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %sew = arith.constant 2 : index
    %lmul = arith.constant 1 : index
    %mask = arith.constant dense<1> : vector<[4]xi1>

    %aRow = memref.dim %arg1, %c0 : memref<?x?xf32>
    %aCol = memref.dim %arg1, %c1 : memref<?x?xf32>
    %bRow = memref.dim %arg2, %c0 : memref<?x?xf32>
    %bCol = memref.dim %arg2, %c1 : memref<?x?xf32>
    affine.for %idx0 = %c0 to %bRow {
      affine.for %idx1 = %c0 to %aRow {
        affine.for %idx2 = %c0 to %aCol {
          %bEle = vector.load %arg1[%idx1, %idx2] : memref<?x?xf32>, vector<[1]xf32>
          %bEle_vector = vector.broadcast %bEle : vector<[1]xf32> to vector<[4]xf32>
          %tmpAVL, %tmpIdx = scf.while (%avl = %bCol, %idx = %c0) : (index, index) -> (index, index) {
            // If avl greater than zero.
            %cond = arith.cmpi sgt, %avl, %c0 : index
            // Pass avl, idx to the after region.
            scf.condition(%cond) %avl, %idx : index, index
          } do {
          ^bb0(%avl : index, %idx : index):
            // Perform the calculation according to the vl.
            %vl = rvv.setvl %avl, %sew, %lmul : index
            %vl_i32 = arith.index_cast %vl : index to i32

            %idx_in_aRow = arith.addi %idx0, %idx1 : index
            %idx_in_aCol = arith.addi %idx2, %idx : index

            %input_vector = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
              %ele = vector.load %arg0[%idx_in_aRow, %idx_in_aCol] : memref<?x?xf32>, vector<[4]xf32>
              vector.yield %ele : vector<[4]xf32>
            } : vector<[4]xf32>
            %c_vector = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
              %ele = vector.load %arg2[%idx0, %idx] : memref<?x?xf32>, vector<[4]xf32>
              vector.yield %ele : vector<[4]xf32>
            } : vector<[4]xf32>
            %result_vector = vector.fma %input_vector, %bEle_vector, %c_vector : vector<[4]xf32>
            vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
              vector.store %result_vector, %arg2[%idx0, %idx] : memref<?x?xf32>, vector<[4]xf32>
              vector.yield
            } : () -> ()
            // Update idx and avl.
            %new_idx = arith.addi %idx, %vl : index
            %new_avl = arith.subi %avl, %vl : index
            scf.yield %new_avl, %new_idx : index, index
          }
        }
      }
    }
    return
  }
}
