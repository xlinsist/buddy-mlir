func.func @generic_op_tensors(
  %arg0 : tensor<64x64x64xf32>, %arg1 : tensor<64x64x64xf32>, %arg2 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %4 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d2, d1, d0)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<64x64x64xf32>, tensor<64x64x64xf32>)
    outs(%arg2 : tensor<64x64x64xf32>) {
    ^bb0(%arg3 : f32, %arg4: f32, %arg5: f32):
      %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
    } -> tensor<64x64x64xf32>
  return %4 : tensor<64x64x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:3 = transform.structured.tile %0 [8, 64, 4] {interchange = [0, 1, 2]} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    transform.structured.vectorize %2
    %b = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap}
    %arg1 {bufferize_function_boundaries = true}
    : (!pdl.operation) -> !pdl.operation
    %f = transform.structured.match ops{["func.func"]} in %b
      : (!pdl.operation) -> !pdl.operation
    %func = transform.vector.lower_contraction %f
      lowering_strategy = "outerproduct"
        : (!pdl.operation) -> !pdl.operation
    %func_1 = transform.vector.transfer_to_scf %func
      max_transfer_rank = 1 full_unroll = false
        : (!pdl.operation) -> !pdl.operation
}

// TODO: Debug why the transformation will raise errors when doing vectorization.
// #map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// func.func @generic_op_tensors(
//   %arg0 : tensor<1x200x128xf32>, %arg1 : tensor<1x200x1xf32>) -> tensor<1x200x1xf32> {
//     %117 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x200x128xf32>) outs(%arg1 : tensor<1x200x1xf32>) {
//     ^bb0(%in: f32, %arg2: f32):
//       %212 = arith.addf %in, %arg2 : f32
//       linalg.yield %212 : f32
//     } -> tensor<1x200x1xf32>
//   return %117 : tensor<1x200x1xf32>
// }

// transform.sequence failures(propagate) {
//   ^bb0(%arg1: !pdl.operation):
//     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
//     // %1, %loops:3 = transform.structured.tile %0 [1, 2, 1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    
//     %1, %loop:2 = transform.structured.tile %0 [1, 1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

//     %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
//     transform.structured.vectorize %2
//     %b = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap}
//     %arg1 {bufferize_function_boundaries = true}
//     : (!pdl.operation) -> !pdl.operation
//     %f = transform.structured.match ops{["func.func"]} in %b
//       : (!pdl.operation) -> !pdl.operation
//     %func = transform.vector.lower_contraction %f
//       lowering_strategy = "outerproduct"
//         : (!pdl.operation) -> !pdl.operation
//     %func_5 = transform.vector.transfer_to_scf %func
//       max_transfer_rank = 1 full_unroll = true
//         : (!pdl.operation) -> !pdl.operation
// }
