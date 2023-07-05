#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ((d0 ceildiv 4) * 4)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
#map4 = affine_map<(d0) -> (d0 + 16)>
#map5 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
#map6 = affine_map<(d0) -> (d0 + 1)>
#map7 = affine_map<(d0, d1) -> (d0 + d1 + 2)>
#map8 = affine_map<(d0) -> (d0 + 2)>
#map9 = affine_map<(d0, d1) -> (d0 + d1 + 3)>
#map10 = affine_map<(d0) -> (d0 + 3)>
#set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>

memref.global "private" @gv_input_f32 : memref<1x64x58x58xf32> = dense<1.0>
memref.global "private" @gv_kernel_f32 : memref<64x64x3x3xf32> = dense<1.0>
memref.global "private" @gv_output_f32 : memref<1x64x56x56xf32>

func.func @main() {

  %arg0 = memref.get_global @gv_input_f32 : memref<1x64x58x58xf32>
  %arg1 = memref.get_global @gv_kernel_f32 : memref<64x64x3x3xf32>
  %arg2 = memref.get_global @gv_output_f32 : memref<1x64x56x56xf32>

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0_0 = arith.constant 0 : index
  %dim = memref.dim %arg2, %c0_0 : memref<1x64x56x56xf32>
  %c1 = arith.constant 1 : index
  %dim_1 = memref.dim %arg2, %c1 : memref<1x64x56x56xf32>
  %c2 = arith.constant 2 : index
  %dim_2 = memref.dim %arg2, %c2 : memref<1x64x56x56xf32>
  %c3 = arith.constant 3 : index
  %dim_3 = memref.dim %arg2, %c3 : memref<1x64x56x56xf32>
  %c1_4 = arith.constant 1 : index
  %dim_5 = memref.dim %arg0, %c1_4 : memref<1x64x58x58xf32>
  %c2_6 = arith.constant 2 : index
  %dim_7 = memref.dim %arg1, %c2_6 : memref<64x64x3x3xf32>
  %c3_8 = arith.constant 3 : index
  %dim_9 = memref.dim %arg1, %c3_8 : memref<64x64x3x3xf32>
  %alloc = memref.alloc() : memref<1xvector<16xf32>>
  affine.for %arg3 = #map(%c0) to #map(%dim) {
    affine.for %arg4 = #map(%c0) to #map(%dim_1) {
      affine.for %arg5 = #map(%c0) to #map(%dim_3) {
        affine.for %arg6 = #map(%c0) to #map(%dim_2) {
          %0 = vector.splat %cst : vector<16xf32>
          memref.store %0, %alloc[%c0] : memref<1xvector<16xf32>>
          affine.for %arg7 = #map(%c0) to #map(%dim_5) {
            %5 = affine.apply #map1(%dim_7)
            affine.for %arg8 = #map(%c0) to #map(%5) step 4 {
              affine.for %arg9 = #map(%c0) to #map(%dim_9) step 32 {
                %6 = affine.apply #map2(%arg6, %arg8)
                %7 = affine.apply #map(%arg8)
                %8 = affine.apply #map2(%arg5, %arg9)
                %9 = affine.apply #map(%arg9)
                %cst_10 = arith.constant 0.000000e+00 : f32
                %10 = vector.transfer_read %arg0[%arg3, %arg7, %6, %8], %cst_10 : memref<1x64x58x58xf32>, vector<16xf32>
                %11 = affine.if #set(%7)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %7, %9], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %12 = affine.apply #map3(%arg5, %arg9)
                %13 = affine.apply #map4(%arg9)
                %cst_11 = arith.constant 0.000000e+00 : f32
                %14 = vector.transfer_read %arg0[%arg3, %arg7, %6, %12], %cst_11 : memref<1x64x58x58xf32>, vector<16xf32>
                %15 = affine.if #set(%7)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %7, %13], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %16 = affine.apply #map5(%arg6, %arg8)
                %17 = affine.apply #map6(%arg8)
                %18 = affine.apply #map2(%arg5, %arg9)
                %19 = affine.apply #map(%arg9)
                %cst_12 = arith.constant 0.000000e+00 : f32
                %20 = vector.transfer_read %arg0[%arg3, %arg7, %16, %18], %cst_12 : memref<1x64x58x58xf32>, vector<16xf32>
                %21 = affine.if #set(%17)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %17, %19], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %22 = affine.apply #map3(%arg5, %arg9)
                %23 = affine.apply #map4(%arg9)
                %cst_13 = arith.constant 0.000000e+00 : f32
                %24 = vector.transfer_read %arg0[%arg3, %arg7, %16, %22], %cst_13 : memref<1x64x58x58xf32>, vector<16xf32>
                %25 = affine.if #set(%17)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %17, %23], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %26 = affine.apply #map7(%arg6, %arg8)
                %27 = affine.apply #map8(%arg8)
                %28 = affine.apply #map2(%arg5, %arg9)
                %29 = affine.apply #map(%arg9)
                %cst_14 = arith.constant 0.000000e+00 : f32
                %30 = vector.transfer_read %arg0[%arg3, %arg7, %26, %28], %cst_14 : memref<1x64x58x58xf32>, vector<16xf32>
                %31 = affine.if #set(%27)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %27, %29], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %32 = affine.apply #map3(%arg5, %arg9)
                %33 = affine.apply #map4(%arg9)
                %cst_15 = arith.constant 0.000000e+00 : f32
                %34 = vector.transfer_read %arg0[%arg3, %arg7, %26, %32], %cst_15 : memref<1x64x58x58xf32>, vector<16xf32>
                %35 = affine.if #set(%27)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %27, %33], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %36 = affine.apply #map9(%arg6, %arg8)
                %37 = affine.apply #map10(%arg8)
                %38 = affine.apply #map2(%arg5, %arg9)
                %39 = affine.apply #map(%arg9)
                %cst_16 = arith.constant 0.000000e+00 : f32
                %40 = vector.transfer_read %arg0[%arg3, %arg7, %36, %38], %cst_16 : memref<1x64x58x58xf32>, vector<16xf32>
                %41 = affine.if #set(%37)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %37, %39], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %42 = affine.apply #map3(%arg5, %arg9)
                %43 = affine.apply #map4(%arg9)
                %cst_17 = arith.constant 0.000000e+00 : f32
                %44 = vector.transfer_read %arg0[%arg3, %arg7, %36, %42], %cst_17 : memref<1x64x58x58xf32>, vector<16xf32>
                %45 = affine.if #set(%37)[%dim_7] -> vector<16xf32> {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %55 = vector.transfer_read %arg1[%arg4, %arg7, %37, %43], %cst_18 : memref<64x64x3x3xf32>, vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                } else {
                  %55 = vector.splat %cst : vector<16xf32>
                  affine.yield %55 : vector<16xf32>
                }
                %46 = memref.load %alloc[%c0] : memref<1xvector<16xf32>>
                %47 = vector.fma %10, %11, %46 : vector<16xf32>
                %48 = vector.fma %14, %15, %47 : vector<16xf32>
                %49 = vector.fma %20, %21, %48 : vector<16xf32>
                %50 = vector.fma %24, %25, %49 : vector<16xf32>
                %51 = vector.fma %30, %31, %50 : vector<16xf32>
                %52 = vector.fma %34, %35, %51 : vector<16xf32>
                %53 = vector.fma %40, %41, %52 : vector<16xf32>
                %54 = vector.fma %44, %45, %53 : vector<16xf32>
                memref.store %54, %alloc[%c0] : memref<1xvector<16xf32>>
              }
            }
          }
          %1 = memref.load %alloc[%c0] : memref<1xvector<16xf32>>
          %2 = vector.reduction <add>, %1 : vector<16xf32> into f32
          %3 = memref.load %arg2[%arg3, %arg4, %arg6, %arg5] : memref<1x64x56x56xf32>
          %4 = arith.addf %3, %2 : f32
          memref.store %4, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<1x64x56x56xf32>
        }
      }
    }
  }
  memref.dealloc %alloc : memref<1xvector<16xf32>>
  %result = vector.load %arg2[%c0, %c0, %c0, %c0] : memref<1x64x56x56xf32>, vector<8xf32>
  vector.print %result : vector<8xf32>
  return
}
