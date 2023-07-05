#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map2 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map3 = affine_map<(d0)[s0] -> (d0 + 2, s0 - 1)>
#map4 = affine_map<(d0)[s0] -> (d0 + 3, s0 - 1)>
#map5 = affine_map<(d0)[s0] -> (d0, s0 - 1)>
#map6 = affine_map<(d0, d1) -> (0)>
#map7 = affine_map<(d0) -> (d0 + 16)>

memref.global "private" @gv_input_f32 : memref<64x64xf32> = dense<1.0>
memref.global "private" @gv_output_f32 : memref<64x64xf32>

func.func @main() {

  %arg0 = memref.get_global @gv_input_f32 : memref<64x64xf32>
  %arg1 = memref.get_global @gv_input_f32 : memref<64x64xf32>
  %arg2 = memref.get_global @gv_output_f32 : memref<64x64xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0_0 : memref<64x64xf32>
  %c1_1 = arith.constant 1 : index
  %dim_2 = memref.dim %arg1, %c1_1 : memref<64x64xf32>
  %c1_3 = arith.constant 1 : index
  %dim_4 = memref.dim %arg0, %c1_3 : memref<64x64xf32>
  affine.for %arg3 = #map(%c0) to #map(%dim_2) step 32 {
    affine.for %arg4 = #map(%c0) to #map(%dim) step 4 {
      %subview = memref.subview %arg0[%arg4, %c0] [%c1, %dim_4] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %0 = affine.min #map2(%arg4)[%dim]
      %subview_5 = memref.subview %arg0[%0, %c0] [%c1, %dim_4] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %1 = affine.min #map3(%arg4)[%dim]
      %subview_6 = memref.subview %arg0[%1, %c0] [%c1, %dim_4] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %2 = affine.min #map4(%arg4)[%dim]
      %subview_7 = memref.subview %arg0[%2, %c0] [%c1, %dim_4] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %3 = affine.min #map5(%arg4)[%dim]
      %subview_8 = memref.subview %arg2[%3, %c0] [%c1, %dim_2] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %4 = affine.min #map2(%arg4)[%dim]
      %subview_9 = memref.subview %arg2[%4, %c0] [%c1, %dim_2] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %5 = affine.min #map3(%arg4)[%dim]
      %subview_10 = memref.subview %arg2[%5, %c0] [%c1, %dim_2] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      %6 = affine.min #map4(%arg4)[%dim]
      %subview_11 = memref.subview %arg2[%6, %c0] [%c1, %dim_2] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, #map1>
      affine.for %arg5 = #map(%c0) to #map(%dim_4) {
        %cst = arith.constant 0.000000e+00 : f32
        %7 = vector.transfer_read %subview[%c0, %arg5], %cst {permutation_map = #map6} : memref<?x?xf32, #map1>, vector<16xf32>
        %cst_12 = arith.constant 0.000000e+00 : f32
        %8 = vector.transfer_read %subview_5[%c0, %arg5], %cst_12 {permutation_map = #map6} : memref<?x?xf32, #map1>, vector<16xf32>
        %cst_13 = arith.constant 0.000000e+00 : f32
        %9 = vector.transfer_read %subview_6[%c0, %arg5], %cst_13 {permutation_map = #map6} : memref<?x?xf32, #map1>, vector<16xf32>
        %cst_14 = arith.constant 0.000000e+00 : f32
        %10 = vector.transfer_read %subview_7[%c0, %arg5], %cst_14 {permutation_map = #map6} : memref<?x?xf32, #map1>, vector<16xf32>
        %11 = affine.apply #map(%arg3)
        %cst_15 = arith.constant 0.000000e+00 : f32
        %12 = vector.transfer_read %subview_8[%c0, %11], %cst_15 : memref<?x?xf32, #map1>, vector<16xf32>
        %13 = affine.apply #map7(%arg3)
        %cst_16 = arith.constant 0.000000e+00 : f32
        %14 = vector.transfer_read %subview_8[%c0, %13], %cst_16 : memref<?x?xf32, #map1>, vector<16xf32>
        %15 = affine.apply #map(%arg3)
        %cst_17 = arith.constant 0.000000e+00 : f32
        %16 = vector.transfer_read %subview_9[%c0, %15], %cst_17 : memref<?x?xf32, #map1>, vector<16xf32>
        %17 = affine.apply #map7(%arg3)
        %cst_18 = arith.constant 0.000000e+00 : f32
        %18 = vector.transfer_read %subview_9[%c0, %17], %cst_18 : memref<?x?xf32, #map1>, vector<16xf32>
        %19 = affine.apply #map(%arg3)
        %cst_19 = arith.constant 0.000000e+00 : f32
        %20 = vector.transfer_read %subview_10[%c0, %19], %cst_19 : memref<?x?xf32, #map1>, vector<16xf32>
        %21 = affine.apply #map7(%arg3)
        %cst_20 = arith.constant 0.000000e+00 : f32
        %22 = vector.transfer_read %subview_10[%c0, %21], %cst_20 : memref<?x?xf32, #map1>, vector<16xf32>
        %23 = affine.apply #map(%arg3)
        %cst_21 = arith.constant 0.000000e+00 : f32
        %24 = vector.transfer_read %subview_11[%c0, %23], %cst_21 : memref<?x?xf32, #map1>, vector<16xf32>
        %25 = affine.apply #map7(%arg3)
        %cst_22 = arith.constant 0.000000e+00 : f32
        %26 = vector.transfer_read %subview_11[%c0, %25], %cst_22 : memref<?x?xf32, #map1>, vector<16xf32>
        %cst_23 = arith.constant 0.000000e+00 : f32
        %27 = vector.transfer_read %arg1[%arg5, %arg3], %cst_23 : memref<64x64xf32>, vector<16xf32>
        %28 = affine.apply #map7(%arg3)
        %cst_24 = arith.constant 0.000000e+00 : f32
        %29 = vector.transfer_read %arg1[%arg5, %28], %cst_24 : memref<64x64xf32>, vector<16xf32>
        %30 = vector.fma %7, %27, %12 : vector<16xf32>
        %31 = vector.fma %7, %29, %14 : vector<16xf32>
        %32 = vector.fma %8, %27, %16 : vector<16xf32>
        %33 = vector.fma %8, %29, %18 : vector<16xf32>
        %34 = vector.fma %9, %27, %20 : vector<16xf32>
        %35 = vector.fma %9, %29, %22 : vector<16xf32>
        %36 = vector.fma %10, %27, %24 : vector<16xf32>
        %37 = vector.fma %10, %29, %26 : vector<16xf32>
        %38 = affine.apply #map(%arg3)
        vector.transfer_write %30, %subview_8[%c0, %38] : vector<16xf32>, memref<?x?xf32, #map1>
        %39 = affine.apply #map7(%arg3)
        vector.transfer_write %31, %subview_8[%c0, %39] : vector<16xf32>, memref<?x?xf32, #map1>
        %40 = affine.apply #map(%arg3)
        vector.transfer_write %32, %subview_9[%c0, %40] : vector<16xf32>, memref<?x?xf32, #map1>
        %41 = affine.apply #map7(%arg3)
        vector.transfer_write %33, %subview_9[%c0, %41] : vector<16xf32>, memref<?x?xf32, #map1>
        %42 = affine.apply #map(%arg3)
        vector.transfer_write %34, %subview_10[%c0, %42] : vector<16xf32>, memref<?x?xf32, #map1>
        %43 = affine.apply #map7(%arg3)
        vector.transfer_write %35, %subview_10[%c0, %43] : vector<16xf32>, memref<?x?xf32, #map1>
        %44 = affine.apply #map(%arg3)
        vector.transfer_write %36, %subview_11[%c0, %44] : vector<16xf32>, memref<?x?xf32, #map1>
        %45 = affine.apply #map7(%arg3)
        vector.transfer_write %37, %subview_11[%c0, %45] : vector<16xf32>, memref<?x?xf32, #map1>
      }
    }
  }
  %result = vector.load %arg2[%c0, %c0] : memref<64x64xf32>, vector<8xf32>
  vector.print %result : vector<8xf32>
  return
}
