// func.func @contraction_batch_matmul(%A: memref<1x12x128xf32>, %B: memref<1x128x128xf32>, %C: memref<1x12x128xf32>) {  linalg.batch_matmul
//     ins(%A, %B: memref<1x12x128xf32>, memref<1x128x128xf32>)
//    outs(%C: memref<1x12x128xf32>)
//   return
// }

// transform.sequence failures(propagate) {
// ^bb1(%arg1: !pdl.operation):
//   %0 = transform.structured.match ops{["linalg.batch_matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
//   %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
//   %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
// }

func.func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}
