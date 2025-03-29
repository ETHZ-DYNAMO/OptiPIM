#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<256x256x3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x13x13xf32> to tensor<1x256x15x15xf32>
    %0 = tensor.empty() : tensor<1x256x13x13xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<256xf32>) outs(%0 : tensor<1x256x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x256x13x13xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst : tensor<1x256x15x15xf32>, tensor<256x256x3x3xf32>) outs(%1 : tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32>
    return %2 : tensor<1x256x13x13xf32>
  }
}
