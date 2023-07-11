// Arrays for conv_2d_nchw_fchw.
memref.global "private" @gv_input_f32 : memref<1x64x58x58xf32> = dense<1.0>
memref.global "private" @gv_kernel_f32 : memref<64x64x3x3xf32> = dense<1.0>
memref.global "private" @gv_output_f32 : memref<1x64x56x56xf32>
// Coefficient matrices for F(6,3).
memref.global "private" @gv_A_f32 : memref<8x6xf32> = dense<[
  [1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
  [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
  [1.000000e+00, -1.000000e+00, 1.000000e+00, -1.000000e+00, 1.000000e+00, -1.000000e+00],
  [1.000000e+00, 2.000000e+00, 4.000000e+00, 8.000000e+00, 1.600000e+01, 3.200000e+01],
  [1.000000e+00, -2.000000e+00, 4.000000e+00, -8.000000e+00, 1.600000e+01, -3.200000e+01],
  [1.000000e+00, 5.000000e-01, 2.500000e-01, 1.250000e-01, 6.250000e-02, 3.125000e-02],
  [1.000000e+00, -5.000000e-01, 2.500000e-01, -1.250000e-01, 6.250000e-02, -3.125000e-02],
  [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]
]>
memref.global "private" @gv_AT_f32 : memref<6x8xf32> = dense<[
  [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, -1.000000e+00, 2.000000e+00, -2.000000e+00, 5.000000e-01, -5.000000e-01, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, 1.000000e+00, 4.000000e+00, 4.000000e+00, 2.500000e-01, 2.500000e-01, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, -1.000000e+00, 8.000000e+00, -8.000000e+00, 1.250000e-01, -1.250000e-01, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.600000e+01, 1.600000e+01, 6.250000e-02, 6.250000e-02, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, -1.000000e+00, 3.200000e+01, -3.200000e+01, 3.125000e-02, -3.125000e-02, 1.000000e+00]
]>
memref.global "private" @gv_B_f32 : memref<8x8xf32> = dense<[
  [1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, -1.000000e+00, 5.000000e-01, -5.000000e-01, 2.000000e+00, -2.000000e+00, -1.000000e+00],
  [-5.250000e+00, 1.000000e+00, 1.000000e+00, 2.500000e-01, 2.500000e-01, 4.000000e+00, 4.000000e+00, 0.000000e+00],
  [0.000000e+00, -4.250000e+00, 4.250000e+00, -1.250000e+00, 1.250000e+00, -1.250000e+00, 1.250000e+00, 5.250000e+00],
  [5.250000e+00, -4.250000e+00, -4.250000e+00, -1.250000e-01, -1.250000e-01, -5.000000e+00, -5.000000e+00, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, -1.000000e+00, 2.000000e+00, 2.000000e+00, 5.000000e-01, -5.000000e-01, -5.250000e+00],
  [-1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]
]>
memref.global "private" @gv_BT_f32 : memref<8x8xf32> = dense<[
  [1.000000e+00, 0.000000e+00, -5.250000e+00, 0.000000e+00, 5.250000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00],
  [0.000000e+00, 1.000000e+00, 1.000000e+00, -4.250000e+00, -4.250000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, -1.000000e+00, 1.000000e+00, 4.250000e+00, -4.250000e+00, -1.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, 5.000000e-01, 2.500000e-01, -1.250000e+00, -1.250000e-01, 2.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, -5.000000e-01, 2.500000e-01, 1.250000e+00, -1.250000e-01, 2.000000e+00, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, 2.000000e+00, 4.000000e+00, -1.250000e+00, -5.000000e+00, 5.000000e-01, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, -2.000000e+00, 4.000000e+00, 1.250000e+00, -5.000000e+00, -5.000000e-01, 1.000000e+00, 0.000000e+00],
  [0.000000e+00, -1.000000e+00, 0.000000e+00, 5.250000e+00, 0.000000e+00, -5.250000e+00, 0.000000e+00, 1.000000e+00]
]>
memref.global "private" @gv_G_f32 : memref<8x3xf32> = dense<[
  [1.000000e+00, 0.000000e+00, 0.000000e+00],
  [-2.222222e-01, -2.222222e-01, -2.222222e-01],
  [-2.222222e-01, 2.222222e-01, -2.222222e-01],
  [1.111111e-02, 2.222222e-02, 4.444444e-02],
  [1.111111e-02, -2.222222e-02, 4.444444e-02],
  [2.222222e-02, 1.111111e-02, 5.555556e-03],
  [2.222222e-02, -1.111111e-02, 5.555556e-03],
  [0.000000e+00, 0.000000e+00, 1.000000e+00]
]>
memref.global "private" @gv_GT_f32 : memref<3x8xf32> = dense<[
  [1.000000e+00, -2.222222e-01, -2.222222e-01, 1.111111e-02, 1.111111e-02, 2.222222e-02, 2.222222e-02, 0.000000e+00],
  [0.000000e+00, -2.222222e-01, 2.222222e-01, 2.222222e-02, -2.222222e-02, 1.111111e-02, -1.111111e-02, 0.000000e+00],
  [0.000000e+00, -2.222222e-01, -2.222222e-01, 4.444444e-02, 4.444444e-02, 5.555556e-03, 5.555556e-03, 1.000000e+00]
]> 

// memref.global "private" @gv_A_f32 : memref<8x6xf32> = dense<[[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]]>
// memref.global "private" @gv_AT_f32 : memref<6x8xf32> = dense<[[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]]>
// memref.global "private" @gv_B_f32 : memref<8x8xf32> = dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 1.0], [-1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0]]>
// memref.global "private" @gv_BT_f32 : memref<8x8xf32> = dense<[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]>
// memref.global "private" @gv_G_f32 : memref<4x3xf32> = dense<[[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]]>
// memref.global "private" @gv_GT_f32 : memref<3x4xf32> = dense<[[1.0, 0.5, 0.5, 0.0], [0.0, 0.5, -0.5, 0.0], [0.0, 0.5, 0.5, 1.0]]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_filled_input_2D(%size0: index, %size1: index) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc(%size0, %size1) : memref<?x?xf32>
  scf.for %idx0 = %c0 to %size0 step %c1 {
    scf.for %idx1 = %c0 to %size1 step %c1 {
    %c0_f32 = arith.constant 0.0 : f32
    memref.store %c0_f32, %mem[%idx0, %idx1] : memref<?x?xf32>
    }
  }
  return %mem : memref<?x?xf32>
}

func.func @kernel_transform(%g: memref<3x3xf32, strided<[3, 1], offset: ?>>, %U : memref<8x8xf32>) {
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  // Initialize coefficient matrices.
  %G = memref.get_global @gv_G_f32 : memref<8x3xf32>
  %GT = memref.get_global @gv_GT_f32 : memref<3x8xf32>
  // Reset intermediate variables.
  %Gg_alloc = func.call @alloc_filled_input_2D(%c8, %c3) : (index, index) -> memref<?x?xf32>
  %Gg = memref.cast %Gg_alloc : memref<?x?xf32> to memref<8x3xf32>
  // Transform kernel: U = GgGT.
  linalg.matmul ins(%G, %g : memref<8x3xf32>, memref<3x3xf32, strided<[3, 1], offset: ?>>) outs(%Gg : memref<8x3xf32>)
  linalg.matmul ins(%Gg, %GT : memref<8x3xf32>, memref<3x8xf32>) outs(%U : memref<8x8xf32>)
  return
}

// result = AT(GgGT*BTdB)A = AT(U*V)A = ATMA
func.func @winagrad_2D(%d: memref<8x8xf32, strided<[58, 1], offset: ?>>, %U: memref<8x8xf32>, %result_mem: memref<6x6xf32, strided<[56, 1], offset: ?>>) {
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  // Initialize coefficient matrices.
  %A = memref.get_global @gv_A_f32 : memref<8x6xf32>
  %AT = memref.get_global @gv_AT_f32 : memref<6x8xf32>
  %B = memref.get_global @gv_B_f32 : memref<8x8xf32>
  %BT = memref.get_global @gv_BT_f32 : memref<8x8xf32>
  // Reset intermediate variables.
  %BTd_alloc = func.call @alloc_filled_input_2D(%c8, %c8) : (index, index) -> memref<?x?xf32>
  %BTd = memref.cast %BTd_alloc : memref<?x?xf32> to memref<8x8xf32>
  %V_alloc = func.call @alloc_filled_input_2D(%c8, %c8) : (index, index) -> memref<?x?xf32>
  %V = memref.cast %V_alloc : memref<?x?xf32> to memref<8x8xf32>
  %M_alloc = func.call @alloc_filled_input_2D(%c8, %c8) : (index, index) -> memref<?x?xf32>
  %M = memref.cast %M_alloc : memref<?x?xf32> to memref<8x8xf32>
  %ATM_alloc = func.call @alloc_filled_input_2D(%c6, %c8) : (index, index) -> memref<?x?xf32>
  %ATM = memref.cast %ATM_alloc : memref<?x?xf32> to memref<6x8xf32>
  // Transform input's tiled unit. V = BTdB.
  linalg.matmul ins(%BT, %d : memref<8x8xf32>, memref<8x8xf32, strided<[58, 1], offset: ?>>) outs(%BTd : memref<8x8xf32>)
  linalg.matmul ins(%BTd, %B : memref<8x8xf32>, memref<8x8xf32>) outs(%V : memref<8x8xf32>)
  // Matrix dot product. M = U*V.
  linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                    affine_map<(d0, d1) -> (d0, d1)>,
                                    affine_map<(d0, d1) -> (d0, d1)>],
                                    iterator_types = ["parallel", "parallel"] }
    ins(%U, %V : memref<8x8xf32>, memref<8x8xf32>)
    outs(%M : memref<8x8xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.mulf %in0, %in1 : f32
      linalg.yield %0 : f32
  }
  // Re-transform output. result = ATMA.
  linalg.matmul ins(%AT, %M : memref<6x8xf32>, memref<8x8xf32>) outs(%ATM : memref<6x8xf32>)
  linalg.matmul ins(%ATM, %A : memref<6x8xf32>, memref<8x6xf32>) outs(%result_mem : memref<6x6xf32, strided<[56, 1], offset: ?>>)
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // Initialize arrays and specify n, f, c, w, h for fixed input and kernel.
  %input = memref.get_global @gv_input_f32 : memref<1x64x58x58xf32>
  %kernel = memref.get_global @gv_kernel_f32 : memref<64x64x3x3xf32>
  %output = memref.get_global @gv_output_f32 : memref<1x64x56x56xf32>
  %n = arith.constant 1 : index
  %f = arith.constant 64 : index
  %c = arith.constant 64 : index
  %input_w = arith.constant 58 : index
  %input_h = arith.constant 58 : index
  // Specify m, r for F(6,3).
  %m = arith.constant 6 : index
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

  affine.for %idx1 = 0 to %f {
    affine.for %idx4 = 0 to %c {
      %g_offset = memref.subview %kernel[%idx1, %idx4, %c0, %c0] [1, 1, 3, 3] [1, 1, 1, 1]: memref<64x64x3x3xf32> to memref<3x3xf32, strided<[3, 1], offset: ?>>
      %U_alloc = func.call @alloc_filled_input_2D(%c8, %c8) : (index, index) -> memref<?x?xf32>
      %U = memref.cast %U_alloc : memref<?x?xf32> to memref<8x8xf32>
      // Preprocess kernel transformation.
      func.call @kernel_transform(%g_offset, %U) : (memref<3x3xf32, strided<[3, 1], offset: ?>>, memref<8x8xf32>) -> ()
      // Implement winograd's minimal filtering algorithm in F(2,3) form.
      affine.for %idx0 = 0 to %n {
        affine.for %idx2 = 0 to %tile_h_num {
          affine.for %idx3 = 0 to %tile_w_num {
            %idx_input_h = index.mul %idx2, %tile_stride
            %idx_input_w = index.mul %idx3, %tile_stride
            %d_offset = memref.subview %input[%idx0, %idx4, %idx_input_h, %idx_input_w] [1, 1, 8, 8] [1, 1, 1, 1]: memref<1x64x58x58xf32> to memref<8x8xf32, strided<[58, 1], offset: ?>>
            %idx_output_h = index.mul %idx2, %m
            %idx_output_w = index.mul %idx3, %m
            %result_mem_offset = memref.subview %output[%idx0, %idx1, %idx_output_h, %idx_output_w] [1, 1, 6, 6] [1, 1, 1, 1]: memref<1x64x56x56xf32> to memref<6x6xf32, strided<[56, 1], offset: ?>>
            func.call @winagrad_2D(%d_offset, %U, %result_mem_offset) : (memref<8x8xf32, strided<[58, 1], offset: ?>>, memref<8x8xf32>, memref<6x6xf32, strided<[56, 1], offset: ?>>) -> ()
            // %converted = memref.cast %result_mem_offset : memref<6x6xf32, strided<[56, 1], offset: ?>> to memref<*xf32>
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
