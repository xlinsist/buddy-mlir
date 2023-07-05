#map = affine_map<(d0) -> (d0 floordiv 9)>
#map1 = affine_map<(d0, d1) -> (d0 floordiv 56 + (d1 mod 9) floordiv 3)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 56) * 56 - (d1 floordiv 3) * 3)>

memref.global "private" @gv_input_f32 : memref<1x64x58x58xf32> = dense<1.0>
memref.global "private" @gv_kernel_f32 : memref<64x64x3x3xf32> = dense<1.0>
memref.global "private" @gv_output_f32 : memref<1x64x56x56xf32>

func.func @main() {
    
  %arg0 = memref.get_global @gv_input_f32 : memref<1x64x58x58xf32>
  %arg1 = memref.get_global @gv_kernel_f32 : memref<64x64x3x3xf32>
  %arg2 = memref.get_global @gv_output_f32 : memref<1x64x56x56xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index // 576 = 64 * 3 * 3 = kernel's f*h*w
  %c3136 = arith.constant 3136 : index // 3136 = 56 * 56 = output's h*w
  %c64 = arith.constant 64 : index
  %kernel_collapse = memref.collapse_shape %arg1 [[0], [1, 2, 3]] : memref<64x64x3x3xf32> into memref<64x576xf32>
  %output_collapse = memref.collapse_shape %arg2 [[0], [1], [2, 3]] : memref<1x64x56x56xf32> into memref<1x64x3136xf32>
  %input_collapse = memref.alloc() {alignment = 64 : i64} : memref<1x576x3136xf32>

  scf.for %idx0 = %c0 to %c1 step %c1 {
    scf.for %idx1 = %c0 to %c576 step %c1 {
      scf.for %idx2 = %c0 to %c3136 step %c1 {
        %0 = affine.apply #map(%idx1)
        %1 = affine.apply #map1(%idx2, %idx1)
        %2 = affine.apply #map2(%idx2, %idx1)
        %3 = memref.load %arg0[%idx0, %0, %1, %2] : memref<1x64x58x58xf32>
        memref.store %3, %input_collapse[%idx0, %idx1, %idx2] : memref<1x576x3136xf32>
      }
    }
  }

  scf.for %idx0 = %c0 to %c1 step %c1 {
    scf.for %idx1 = %c0 to %c64 step %c1 {
      scf.for %idx2 = %c0 to %c3136 step %c1 {
        scf.for %idx3 = %c0 to %c576 step %c1 {
          %0 = memref.load %kernel_collapse[%idx1, %idx3] : memref<64x576xf32>
          %1 = memref.load %input_collapse[%idx0, %idx3, %idx2] : memref<1x576x3136xf32>
          %2 = memref.load %output_collapse[%idx0, %idx1, %idx2] : memref<1x64x3136xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %3, %2 : f32
          memref.store %4, %output_collapse[%idx0, %idx1, %idx2] : memref<1x64x3136xf32>
        }
      }
    }
  }
  memref.dealloc %input_collapse : memref<1x576x3136xf32>
  %result_mem = memref.expand_shape %output_collapse [[0], [1], [2, 3]] : memref<1x64x3136xf32> into memref<1x64x56x56xf32>
  
  %result = vector.load %result_mem[%c0, %c0, %c0, %c0] : memref<1x64x56x56xf32>, vector<8xf32>
  vector.print %result : vector<8xf32>
  return
}
