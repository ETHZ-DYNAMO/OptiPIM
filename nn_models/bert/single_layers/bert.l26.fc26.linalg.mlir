#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @batch_matmul_12x128x64x128(%arg0: tensor<12x128x64xf32>, %arg1: tensor<12x64x128xf32>) -> tensor<12x128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<12x128x128xf32>

  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg1 : tensor<12x64x128xf32>) 
    outs(%arg1 : tensor<12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<12x64x128xf32>

  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>

  %3 = linalg.batch_matmul 
    ins(%arg0, %1 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) 
    outs(%2 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>

  return %3 : tensor<12x128x128xf32>
}