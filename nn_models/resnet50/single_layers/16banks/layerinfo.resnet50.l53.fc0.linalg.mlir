
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "Linear"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x2048xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1000x2048xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<1000xf32>
    %0 = tensor.empty() : tensor<2048x1000xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<1000x2048xf32>) outs(%0 : tensor<2048x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2048x1000xf32>
    %2 = tensor.empty() : tensor<1x1000xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %4 = linalg.matmul {global_layer_idx = 0 : i32, layer_group = 0 : i32, num_banks = 16 : i32, type_layer_idx = 0 : i32} ins(%arg0, %1 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%3 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%2 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<1x1000xf32>
    return %5 : tensor<1x1000xf32>
  }
}
