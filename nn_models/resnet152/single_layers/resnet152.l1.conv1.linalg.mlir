module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x64x1x1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64x56x56xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) outs(%1 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %2 : tensor<1x64x56x56xf32>
  }
}
