
module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x32x224x224xf32>) -> tensor<1x32x224x224xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x32x3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x32x224x224xf32> to tensor<1x32x226x226xf32>
    %0 = tensor.empty() : tensor<1x32x224x224xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x32x224x224xf32>) -> tensor<1x32x224x224xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, global_layer_idx = 0 : i32, layer_group = 0 : i32, num_banks = 16 : i32, strides = dense<1> : vector<2xi64>, type_layer_idx = 0 : i32} ins(%padded, %cst : tensor<1x32x226x226xf32>, tensor<32x32x3x3xf32>) outs(%1 : tensor<1x32x224x224xf32>) -> tensor<1x32x224x224xf32>
    return %2 : tensor<1x32x224x224xf32>
  }
}
