// Arrays for conv_2d_nchw_fchw.
memref.global "private" @gv_input_f32 : memref<1x64x58x58xf32> = dense<1.0>
memref.global "private" @gv_kernel_f32 : memref<64x64x3x3xf32> = dense<1.0>
memref.global "private" @gv_output_f32 : memref<1x64x56x56xf32>
// Coefficient matrices for F(2,3).
memref.global "private" @gv_A_f32 : memref<4x2xf32> = dense<[[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]]>
memref.global "private" @gv_AT_f32 : memref<2x4xf32> = dense<[[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]]>
memref.global "private" @gv_B_f32 : memref<4x4xf32> = dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 1.0], [-1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0]]>
memref.global "private" @gv_BT_f32 : memref<4x4xf32> = dense<[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]>
memref.global "private" @gv_G_f32 : memref<4x3xf32> = dense<[[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]]>
memref.global "private" @gv_GT_f32 : memref<3x4xf32> = dense<[[1.0, 0.5, 0.5, 0.0], [0.0, 0.5, -0.5, 0.0], [0.0, 0.5, 0.5, 1.0]]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_filled_input_2D(%size0: index, %size1: index) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc(%size0, %size1) : memref<?x?xf32>
  scf.for %idx2 = %c0 to %size0 step %c1 {
    scf.for %idx0 = %c0 to %size1 step %c1 {
    %c0_f32 = arith.constant 0.0 : f32
    memref.store %c0_f32, %mem[%idx2, %idx0] : memref<?x?xf32>
    }
  }
  return %mem : memref<?x?xf32>
}

func.func @kernel_transform(%g: memref<3x3xf32, strided<[3, 1], offset: ?>>, %U : memref<4x4xf32>) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // Initialize coefficient matrices.
  %G = memref.get_global @gv_G_f32 : memref<4x3xf32>
  %GT = memref.get_global @gv_GT_f32 : memref<3x4xf32>
  // Reset intermediate variables.
  %Gg_alloc = func.call @alloc_filled_input_2D(%c4, %c3) : (index, index) -> memref<?x?xf32>
  %Gg = memref.cast %Gg_alloc : memref<?x?xf32> to memref<4x3xf32>
  // Transform kernel: U = GgGT.
  linalg.matmul ins(%G, %g : memref<4x3xf32>, memref<3x3xf32, strided<[3, 1], offset: ?>>) outs(%Gg : memref<4x3xf32>)
  linalg.matmul ins(%Gg, %GT : memref<4x3xf32>, memref<3x4xf32>) outs(%U : memref<4x4xf32>)
  return
}

// result = AT(GgGT*BTdB)A = AT(U*V)A = ATMA
func.func @winagrad_2D(%d: memref<4x4xf32, strided<[58, 1], offset: ?>>, %U: memref<4x4xf32>, %result_mem: memref<2x2xf32, strided<[56, 1], offset: ?>>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // Initialize coefficient matrices.
  %A = memref.get_global @gv_A_f32 : memref<4x2xf32>
  %AT = memref.get_global @gv_AT_f32 : memref<2x4xf32>
  %B = memref.get_global @gv_B_f32 : memref<4x4xf32>
  %BT = memref.get_global @gv_BT_f32 : memref<4x4xf32>
  // Reset intermediate variables.
  %BTd_alloc = func.call @alloc_filled_input_2D(%c4, %c4) : (index, index) -> memref<?x?xf32>
  %BTd = memref.cast %BTd_alloc : memref<?x?xf32> to memref<4x4xf32>
  %V_alloc = func.call @alloc_filled_input_2D(%c4, %c4) : (index, index) -> memref<?x?xf32>
  %V = memref.cast %V_alloc : memref<?x?xf32> to memref<4x4xf32>
  %M_alloc = func.call @alloc_filled_input_2D(%c4, %c4) : (index, index) -> memref<?x?xf32>
  %M = memref.cast %M_alloc : memref<?x?xf32> to memref<4x4xf32>
  %ATM_alloc = func.call @alloc_filled_input_2D(%c2, %c4) : (index, index) -> memref<?x?xf32>
  %ATM = memref.cast %ATM_alloc : memref<?x?xf32> to memref<2x4xf32>
  // Transform input's tiled unit. V = BTdB.
  linalg.matmul ins(%BT, %d : memref<4x4xf32>, memref<4x4xf32, strided<[58, 1], offset: ?>>) outs(%BTd : memref<4x4xf32>)
  linalg.matmul ins(%BTd, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%V : memref<4x4xf32>)
  // Matrix dot product. M = U*V.
  linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                    affine_map<(d0, d1) -> (d0, d1)>,
                                    affine_map<(d0, d1) -> (d0, d1)>],
                                    iterator_types = ["parallel", "parallel"] }
    ins(%U, %V : memref<4x4xf32>, memref<4x4xf32>)
    outs(%M : memref<4x4xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.mulf %in0, %in1 : f32
      linalg.yield %0 : f32
  }
  // Re-transform output. result = ATMA.
  linalg.matmul ins(%AT, %M : memref<2x4xf32>, memref<4x4xf32>) outs(%ATM : memref<2x4xf32>)
  linalg.matmul ins(%ATM, %A : memref<2x4xf32>, memref<4x2xf32>) outs(%result_mem : memref<2x2xf32, strided<[56, 1], offset: ?>>)
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // Initialize arrays and specify n, f, c, w, h for fixed input and kernel.
  %input = memref.get_global @gv_input_f32 : memref<1x64x58x58xf32>
  %kernel = memref.get_global @gv_kernel_f32 : memref<64x64x3x3xf32>
  %output = memref.get_global @gv_output_f32 : memref<1x64x56x56xf32>
  %n = arith.constant 1 : index
  %f = arith.constant 64 : index
  %c = arith.constant 64 : index
  %input_w = arith.constant 58 : index
  %input_h = arith.constant 58 : index
  // Specify m, r for F(2,3).
  %m = arith.constant 2 : index
  %r = arith.constant 3 : index
  // Input's tiled unit size, equal to m + r - 1.
  %tile_size_plus = index.add %m, %r
  %tile_size = index.sub %tile_size_plus, %c1
  // Stride while tiling, equal to tile_size - (r - 1).
  %tile_overlap = index.sub %r, %c1
  %tile_stride = index.sub %tile_size, %tile_overlap
  // Number of tiled unit size along the width axis, equal to (input_w - tile_size) / tile_stride + 1. 
  %tile_w_remain = index.sub %input_w, %tile_size
  %tile_w_num_minus = index.ceildivu %tile_w_remain, %tile_stride
  %tile_w_num = index.add %tile_w_num_minus, %c1
  // Number of tiled unit size along the height axis, equal to (input_h - tile_size) / tile_stride + 1. 
  %tile_h_remain = index.sub %input_h, %tile_size
  %tile_h_num_minus = index.ceildivu %tile_h_remain, %tile_stride
  %tile_h_num = index.add %tile_h_num_minus, %c1

  affine.for %idx0 = 0 to %f {
    affine.for %idx1 = 0 to %c {
      %g_offset = memref.subview %kernel[%idx0, %idx1, %c0, %c0] [1, 1, 3, 3] [1, 1, 1, 1]: memref<64x64x3x3xf32> to memref<3x3xf32, strided<[3, 1], offset: ?>>
      %U_alloc = func.call @alloc_filled_input_2D(%c4, %c4) : (index, index) -> memref<?x?xf32>
      %U = memref.cast %U_alloc : memref<?x?xf32> to memref<4x4xf32>
      // Preprocess kernel transformation.
      func.call @kernel_transform(%g_offset, %U) : (memref<3x3xf32, strided<[3, 1], offset: ?>>, memref<4x4xf32>) -> ()
      // Implement winograd's minimal filtering algorithm in F(2,3) form.
      affine.for %idx2 = 0 to %n {
        affine.for %idx3 = 0 to %tile_h_num {
          affine.for %idx4 = 0 to %tile_w_num {
            %idx_input_h = index.mul %idx3, %tile_stride
            %idx_input_w = index.mul %idx4, %tile_stride
            %d_offset = memref.subview %input[%idx2, %idx1, %idx_input_h, %idx_input_w] [1, 1, 4, 4] [1, 1, 1, 1]: memref<1x64x58x58xf32> to memref<4x4xf32, strided<[58, 1], offset: ?>>
            %idx_output_h = index.mul %idx3, %m
            %idx_output_w = index.mul %idx4, %m
            %result_mem_offset = memref.subview %output[%idx2, %idx0, %idx_output_h, %idx_output_w] [1, 1, 2, 2] [1, 1, 1, 1]: memref<1x64x56x56xf32> to memref<2x2xf32, strided<[56, 1], offset: ?>>
            func.call @winagrad_2D(%d_offset, %U, %result_mem_offset) : (memref<4x4xf32, strided<[58, 1], offset: ?>>, memref<4x4xf32>, memref<2x2xf32, strided<[56, 1], offset: ?>>) -> ()
            // %converted = memref.cast %result_mem_offset : memref<2x2xf32, strided<[56, 1], offset: ?>> to memref<*xf32>
            // func.call @printMemrefF32(%converted): (memref<*xf32>) -> ()
          }
        }
      }
    }
  }
  %result = vector.load %output[%c0, %c1, %c1, %c0] : memref<1x64x56x56xf32>, vector<56xf32>
  vector.print %result : vector<56xf32>
  return
}
