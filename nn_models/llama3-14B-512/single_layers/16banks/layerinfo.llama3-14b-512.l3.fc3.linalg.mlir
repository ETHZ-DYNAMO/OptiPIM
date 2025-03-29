module attributes {torch.debug_module_name = "ModelingMatmul"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<32x512x128xf32>) -> tensor<32x512x512xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x128x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x128x512xf32>) -> tensor<32x128x512xf32>
    %2 = tensor.empty() : tensor<32x512x512xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x512x512xf32>) -> tensor<32x512x512xf32>
    %4 = linalg.batch_matmul {global_layer_idx = 0 : i32, layer_group = 0 : i32, num_banks = 16 : i32, type_layer_idx = 0 : i32} ins(%arg0, %1 : tensor<32x512x128xf32>, tensor<32x128x512xf32>) outs(%3 : tensor<32x512x512xf32>) -> tensor<32x512x512xf32>
    return %4 : tensor<32x512x512xf32>
  }
}
