
#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x32x224x224xf32>) -> tensor<1x1x224x224xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x32x1x1xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %0 = tensor.empty() : tensor<1x1x224x224xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<1xf32>) outs(%0 : tensor<1x1x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x224x224xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, global_layer_idx = 0 : i32, layer_group = 0 : i32, num_banks = 16 : i32, strides = dense<1> : vector<2xi64>, type_layer_idx = 0 : i32} ins(%arg0, %cst : tensor<1x32x224x224xf32>, tensor<1x32x1x1xf32>) outs(%1 : tensor<1x1x224x224xf32>) -> tensor<1x1x224x224xf32>
    return %2 : tensor<1x1x224x224xf32>
  }
}
