
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @batch_matmul_1x128x3072x768(%arg0: tensor<1x128x3072xf32>, %arg1: tensor<1x3072x768xf32>) -> tensor<1x128x768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x3072x768xf32>) outs(%arg1 : tensor<1x3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3072x768xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %3 = linalg.batch_matmul {global_layer_idx = 0 : i32, layer_group = 0 : i32, num_banks = 16 : i32, type_layer_idx = 0 : i32} ins(%arg0, %1 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%2 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    return %3 : tensor<1x128x768xf32>
  }
}
