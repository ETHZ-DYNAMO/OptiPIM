module attributes {torch.debug_module_name = "ResNet"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x3x7x7xf32>
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<64x64x1x1xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<64x64x3x3xf32>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<256x64x1x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<64x256x1x1xf32>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<128x256x1x1xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_13 = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_14 = arith.constant dense<1.000000e+00> : tensor<128x128x3x3xf32>
    %cst_15 = arith.constant dense<1.000000e+00> : tensor<512x128x1x1xf32>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_17 = arith.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_18 = arith.constant dense<1.000000e+00> : tensor<512x256x1x1xf32>
    %cst_19 = arith.constant dense<1.000000e+00> : tensor<128x512x1x1xf32>
    %cst_20 = arith.constant dense<1.000000e+00> : tensor<256x512x1x1xf32>
    %cst_21 = arith.constant dense<1.000000e+00> : tensor<256x256x3x3xf32>
    %cst_22 = arith.constant dense<1.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_23 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_24 = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %cst_25 = arith.constant dense<1.000000e+00> : tensor<1024x512x1x1xf32>
    %cst_26 = arith.constant dense<1.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_27 = arith.constant dense<1.000000e+00> : tensor<512x1024x1x1xf32>
    %cst_28 = arith.constant dense<1.000000e+00> : tensor<512x512x3x3xf32>
    %cst_29 = arith.constant dense<1.000000e+00> : tensor<2048x512x1x1xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<2048xf32>
    %cst_31 = arith.constant dense<1.000000e+00> : tensor<2048xf32>
    %cst_32 = arith.constant dense<1.000000e+00> : tensor<2048x1024x1x1xf32>
    %cst_33 = arith.constant dense<1.000000e+00> : tensor<512x2048x1x1xf32>
    %cst_34 = arith.constant dense<1.000000e+00> : tensor<1000x2048xf32>
    %cst_35 = arith.constant dense<1.000000e+00> : tensor<1000xf32>
    %padded = tensor.pad %arg0 low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
    %0 = tensor.empty() : tensor<1x64x112x112xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %cst : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%1 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%2 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x112x112xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x112x112xf32>
    %padded_36 = tensor.pad %4 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %5 = tensor.empty() : tensor<1x64x56x56xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %7 = tensor.empty() : tensor<3x3xf32>
    %8 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_36, %7 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%6 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%8, %cst_5 : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_37 = tensor.pad %12 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_37, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%13 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %16 = tensor.empty() : tensor<1x256x56x56xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %18 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%15, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%18 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x56x56xf32>
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%8, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%20 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x56x56xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %21 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x256x56x56xf32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x56x56xf32>
    %24 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%23, %cst_10 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%24 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_38 = tensor.pad %26 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %27 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_38, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%27 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %30 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%29, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%30 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x56x56xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %23 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x256x56x56xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x56x56xf32>
    %34 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%33, %cst_10 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%34 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_39 = tensor.pad %36 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %37 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_39, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%37 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x64x56x56xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x64x56x56xf32>
    %40 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%39, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%40 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x56x56xf32>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %33 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x256x56x56xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x56x56xf32>
    %44 = tensor.empty() : tensor<1x128x56x56xf32>
    %45 = linalg.fill ins(%cst_0 : f32) outs(%44 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %46 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%43, %cst_11 : tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) outs(%45 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x56x56xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%46 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x56x56xf32>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<1x128x56x56xf32>) outs(%44 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x56x56xf32>
    %padded_40 = tensor.pad %48 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
    %49 = tensor.empty() : tensor<1x128x28x28xf32>
    %50 = linalg.fill ins(%cst_0 : f32) outs(%49 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %51 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_40, %cst_14 : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%51 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %54 = tensor.empty() : tensor<1x512x28x28xf32>
    %55 = linalg.fill ins(%cst_0 : f32) outs(%54 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %56 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%53, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%56 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x28x28xf32>
    %58 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%43, %cst_18 : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%58 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x28x28xf32>
    %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %59 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x512x28x28xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x28x28xf32>
    %62 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%61, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%62 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_41 = tensor.pad %64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %65 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_41, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%65 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %68 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%67, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%68 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x28x28xf32>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %61 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x512x28x28xf32>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x28x28xf32>
    %72 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%71, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%72 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_42 = tensor.pad %74 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %75 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_42, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%75 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %78 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%77, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%78 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x28x28xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %71 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x512x28x28xf32>
    %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x28x28xf32>
    %82 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%81, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%82 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_43 = tensor.pad %84 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %85 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_43, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%85 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x128x28x28xf32>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x128x28x28xf32>
    %88 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%87, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%88 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x28x28xf32>
    %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %81 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x512x28x28xf32>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x28x28xf32>
    %92 = tensor.empty() : tensor<1x256x28x28xf32>
    %93 = linalg.fill ins(%cst_0 : f32) outs(%92 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %94 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%91, %cst_20 : tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) outs(%93 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x28x28xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%94 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x28x28xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95 : tensor<1x256x28x28xf32>) outs(%92 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x28x28xf32>
    %padded_44 = tensor.pad %96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x28x28xf32> to tensor<1x256x30x30xf32>
    %97 = tensor.empty() : tensor<1x256x14x14xf32>
    %98 = linalg.fill ins(%cst_0 : f32) outs(%97 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %99 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_44, %cst_21 : tensor<1x256x30x30xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%99 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %102 = tensor.empty() : tensor<1x1024x14x14xf32>
    %103 = linalg.fill ins(%cst_0 : f32) outs(%102 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %104 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%101, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%104 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %106 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%91, %cst_25 : tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%106 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %107 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %110 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%109, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%110 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_45 = tensor.pad %112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %113 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_45, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%113 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %116 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%115, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%116 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %109 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %120 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%119, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%120 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_46 = tensor.pad %122 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %123 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_46, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%123 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %126 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%125, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%126 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%127, %119 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%128 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %130 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%129, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%130 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_47 = tensor.pad %132 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %133 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_47, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%133 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %136 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%135, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %137 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%136 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%137, %129 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%138 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %140 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%139, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%140 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%141 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_48 = tensor.pad %142 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %143 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_48, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%143, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%143 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %146 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%145, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%146 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %139 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %150 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%149, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%150 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_49 = tensor.pad %152 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %153 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_49, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%98 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%153 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x256x14x14xf32>
    %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154 : tensor<1x256x14x14xf32>) outs(%97 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x256x14x14xf32>
    %156 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%155, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%103 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%156 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x1024x14x14xf32>
    %158 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%157, %149 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1024x14x14xf32>
    %159 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%158 : tensor<1x1024x14x14xf32>) outs(%102 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x1024x14x14xf32>
    %160 = tensor.empty() : tensor<1x512x14x14xf32>
    %161 = linalg.fill ins(%cst_0 : f32) outs(%160 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %162 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%159, %cst_27 : tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) outs(%161 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x14x14xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%162 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x14x14xf32>
    %164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163 : tensor<1x512x14x14xf32>) outs(%160 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_50 = tensor.pad %164 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %165 = tensor.empty() : tensor<1x512x7x7xf32>
    %166 = linalg.fill ins(%cst_0 : f32) outs(%165 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %167 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_50, %cst_28 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%166 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%167, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%167 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x7x7xf32>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168 : tensor<1x512x7x7xf32>) outs(%165 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x7x7xf32>
    %170 = tensor.empty() : tensor<1x2048x7x7xf32>
    %171 = linalg.fill ins(%cst_0 : f32) outs(%170 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %172 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%169, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%171 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%172 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x2048x7x7xf32>
    %174 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%159, %cst_32 : tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) outs(%171 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %175 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%174 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x2048x7x7xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%173, %175 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x2048x7x7xf32>
    %177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176 : tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x2048x7x7xf32>
    %178 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%177, %cst_33 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%166 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%178 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x7x7xf32>
    %180 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%179 : tensor<1x512x7x7xf32>) outs(%165 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_51 = tensor.pad %180 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %181 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_51, %cst_28 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%166 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%181 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x7x7xf32>
    %183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%182 : tensor<1x512x7x7xf32>) outs(%165 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x7x7xf32>
    %184 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%183, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%171 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%184 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x2048x7x7xf32>
    %186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %177 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x2048x7x7xf32>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186 : tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x2048x7x7xf32>
    %188 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%187, %cst_33 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%166 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%188 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x7x7xf32>
    %190 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189 : tensor<1x512x7x7xf32>) outs(%165 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_52 = tensor.pad %190 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %191 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_52, %cst_28 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%166 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%191 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x512x7x7xf32>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%192 : tensor<1x512x7x7xf32>) outs(%165 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x512x7x7xf32>
    %194 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%193, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%171 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%194, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%194 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %in_55: f32, %in_56: f32, %in_57: f32, %out: f32):
      %210 = arith.truncf %cst_2 : f64 to f32
      %211 = arith.addf %in_57, %210 : f32
      %212 = math.rsqrt %211 : f32
      %213 = arith.subf %in, %in_56 : f32
      %214 = arith.mulf %213, %212 : f32
      %215 = arith.mulf %214, %in_54 : f32
      %216 = arith.addf %215, %in_55 : f32
      linalg.yield %216 : f32
    } -> tensor<1x2048x7x7xf32>
    %196 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %187 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x2048x7x7xf32>
    %197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%196 : tensor<1x2048x7x7xf32>) outs(%170 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.cmpf ugt, %in, %cst_0 : f32
      %211 = arith.select %210, %in, %cst_0 : f32
      linalg.yield %211 : f32
    } -> tensor<1x2048x7x7xf32>
    %198 = tensor.empty() : tensor<7x7xi1>
    %199 = tensor.empty() : tensor<1x1xf32>
    %200 = tensor.empty() : tensor<1x2048x1x1xf32>
    %201 = linalg.fill ins(%cst_0 : f32) outs(%200 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %padded_53 = tensor.pad %197 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x2048x7x7xf32> to tensor<1x2048x8x8xf32>
    %202:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%198 : tensor<7x7xi1>) outs(%201, %199 : tensor<1x2048x1x1xf32>, tensor<1x1xf32>) {
    ^bb0(%in: i1, %out: f32, %out_54: f32):
      %210 = linalg.index 0 : index
      %211 = linalg.index 1 : index
      %212 = linalg.index 2 : index
      %213 = linalg.index 3 : index
      %214 = linalg.index 4 : index
      %215 = linalg.index 5 : index
      %216 = arith.muli %212, %c7 : index
      %217 = arith.addi %212, %c1 : index
      %218 = arith.muli %217, %c7 : index
      %219 = arith.muli %213, %c7 : index
      %220 = arith.addi %213, %c1 : index
      %221 = arith.muli %220, %c7 : index
      %222 = arith.addi %216, %214 : index
      %223 = arith.addi %219, %215 : index
      %extracted = tensor.extract %padded_53[%210, %211, %222, %223] : tensor<1x2048x8x8xf32>
      %224 = arith.cmpi ult, %222, %218 : index
      %225 = arith.select %224, %extracted, %cst_0 : f32
      %226 = arith.cmpi ult, %223, %221 : index
      %227 = arith.select %226, %225, %cst_0 : f32
      %228 = arith.addf %227, %out : f32
      %229 = arith.subi %218, %216 : index
      %230 = arith.subi %221, %219 : index
      %231 = arith.muli %229, %230 : index
      %232 = arith.index_cast %231 : index to i64
      %233 = arith.sitofp %232 : i64 to f32
      linalg.yield %228, %233 : f32, f32
    } -> (tensor<1x2048x1x1xf32>, tensor<1x1xf32>)
    %203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%202#1 : tensor<1x1xf32>) outs(%202#0 : tensor<1x2048x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %210 = arith.divf %out, %in : f32
      linalg.yield %210 : f32
    } -> tensor<1x2048x1x1xf32>
    %collapsed = tensor.collapse_shape %203 [[0], [1, 2, 3]] : tensor<1x2048x1x1xf32> into tensor<1x2048xf32>
    %204 = tensor.empty() : tensor<2048x1000xf32>
    %205 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_34 : tensor<1000x2048xf32>) outs(%204 : tensor<2048x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2048x1000xf32>
    %206 = tensor.empty() : tensor<1x1000xf32>
    %207 = linalg.fill ins(%cst_0 : f32) outs(%206 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %208 = linalg.matmul ins(%collapsed, %205 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%207 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%208, %cst_35 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%206 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_54: f32, %out: f32):
      %210 = arith.addf %in, %in_54 : f32
      linalg.yield %210 : f32
    } -> tensor<1x1000xf32>
    return %209 : tensor<1x1000xf32>
  }
}
