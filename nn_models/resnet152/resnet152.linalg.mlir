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
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x112x112xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
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
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_37 = tensor.pad %12 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_37, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%13 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %16 = tensor.empty() : tensor<1x256x56x56xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %18 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%15, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%18 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x56x56xf32>
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%8, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%20 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x56x56xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %21 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x256x56x56xf32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x56x56xf32>
    %24 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%23, %cst_10 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%24 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_38 = tensor.pad %26 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %27 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_38, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%27 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %30 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%29, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%30 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x56x56xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %23 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x256x56x56xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x56x56xf32>
    %34 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%33, %cst_10 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%34 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_39 = tensor.pad %36 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %37 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_39, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %cst_4, %cst_4, %cst_3, %cst_4 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%37 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x64x56x56xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x64x56x56xf32>
    %40 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%39, %cst_7 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%17 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%40 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x56x56xf32>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %33 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x256x56x56xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<1x256x56x56xf32>) outs(%16 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x56x56xf32>
    %44 = tensor.empty() : tensor<1x128x56x56xf32>
    %45 = linalg.fill ins(%cst_0 : f32) outs(%44 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %46 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%43, %cst_11 : tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) outs(%45 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x56x56xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%46 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x56x56xf32>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<1x128x56x56xf32>) outs(%44 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x56x56xf32>
    %padded_40 = tensor.pad %48 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
    %49 = tensor.empty() : tensor<1x128x28x28xf32>
    %50 = linalg.fill ins(%cst_0 : f32) outs(%49 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %51 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_40, %cst_14 : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%51 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %54 = tensor.empty() : tensor<1x512x28x28xf32>
    %55 = linalg.fill ins(%cst_0 : f32) outs(%54 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %56 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%53, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%56 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %58 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%43, %cst_18 : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%58 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %59 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %62 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%61, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%62 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_41 = tensor.pad %64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %65 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_41, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%65 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %68 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%67, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%68 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %61 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %72 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%71, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%72 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_42 = tensor.pad %74 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %75 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_42, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%75 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %78 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%77, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%78 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %71 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %82 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%81, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%82 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_43 = tensor.pad %84 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %85 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_43, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%85 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %88 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%87, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%88 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %81 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %92 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%91, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%92 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_44 = tensor.pad %94 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %95 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_44, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%95 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %98 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%97, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%98 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %91 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %102 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%101, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%102 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_45 = tensor.pad %104 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %105 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_45, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%105 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %108 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%107, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%108 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %101 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %112 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%111, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%112 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_46 = tensor.pad %114 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %115 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_46, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%115 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %118 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%117, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%118 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %111 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %122 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%121, %cst_19 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%122 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_47 = tensor.pad %124 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %125 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_47, %cst_14 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%50 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %cst_13, %cst_13, %cst_12, %cst_13 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%125 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x128x28x28xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126 : tensor<1x128x28x28xf32>) outs(%49 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x128x28x28xf32>
    %128 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%127, %cst_15 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%55 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%128, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%128 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x28x28xf32>
    %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %121 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x512x28x28xf32>
    %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130 : tensor<1x512x28x28xf32>) outs(%54 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x28x28xf32>
    %132 = tensor.empty() : tensor<1x256x28x28xf32>
    %133 = linalg.fill ins(%cst_0 : f32) outs(%132 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %134 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%131, %cst_20 : tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) outs(%133 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x28x28xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%134 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x28x28xf32>
    %136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%135 : tensor<1x256x28x28xf32>) outs(%132 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x28x28xf32>
    %padded_48 = tensor.pad %136 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x28x28xf32> to tensor<1x256x30x30xf32>
    %137 = tensor.empty() : tensor<1x256x14x14xf32>
    %138 = linalg.fill ins(%cst_0 : f32) outs(%137 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %139 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_48, %cst_21 : tensor<1x256x30x30xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%139, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%139 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %142 = tensor.empty() : tensor<1x1024x14x14xf32>
    %143 = linalg.fill ins(%cst_0 : f32) outs(%142 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %144 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%141, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%144 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %146 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%131, %cst_25 : tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%146 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %147 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %150 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%149, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%150 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_49 = tensor.pad %152 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %153 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_49, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%153 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %156 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%155, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%156 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %158 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%157, %149 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %159 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%158 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %160 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%159, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %161 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%160, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%160 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%161 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_50 = tensor.pad %162 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %163 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_50, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%163 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%164 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %166 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%165, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %167 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%166, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%166 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%167, %159 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %170 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%169, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%170, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%170 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%171 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_51 = tensor.pad %172 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %173 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_51, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%173, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%173 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %175 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %176 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%175, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%176 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%177, %169 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %180 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%179, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%180 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_52 = tensor.pad %182 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %183 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_52, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%183, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%183 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %186 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%185, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%186 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%187, %179 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %190 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%189, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %191 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%190, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%190 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_53 = tensor.pad %192 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %193 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_53, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%193 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%194 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %196 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%195, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%196, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%196 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %189 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%198 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %200 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%199, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%200, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%200 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%201 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_54 = tensor.pad %202 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %203 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_54, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%203, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%203 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %205 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%204 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %206 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%205, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%206 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%207, %199 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%208 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %210 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%209, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%210 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%211 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_55 = tensor.pad %212 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %213 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_55, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%213, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%213 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%214 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %216 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%215, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%216, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%216 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217, %209 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %219 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%218 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %220 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%219, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%220, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%220 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %222 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%221 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_56 = tensor.pad %222 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %223 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_56, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%223, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%223 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%224 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %226 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%225, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%226 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%227, %219 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %230 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%229, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%230 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %232 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%231 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_57 = tensor.pad %232 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %233 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_57, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%233, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%233 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %236 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%235, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%236, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%236 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %238 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%237, %229 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%238 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %240 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%239, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%240, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%240 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_58 = tensor.pad %242 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %243 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_58, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%243 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %246 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%245, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%246, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%246 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%247, %239 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%248 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %250 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%249, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%250, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%250 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_59 = tensor.pad %252 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %253 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_59, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %254 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%253, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%253 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %256 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%255, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %257 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%256 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%257, %249 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%258 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %260 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%259, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%260 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_60 = tensor.pad %262 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %263 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_60, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %264 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%263, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%263 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%264 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %266 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%265, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%266, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%266 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%267, %259 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %270 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%269, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%270, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%270 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %272 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%271 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_61 = tensor.pad %272 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %273 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_61, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%273, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%273 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %276 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%275, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%276 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %278 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%277, %269 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%278 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %280 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%279, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%280, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%280 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%281 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_62 = tensor.pad %282 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %283 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_62, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%283, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%283 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %286 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%285, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%286, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%286 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %279 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %290 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%289, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%290 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_63 = tensor.pad %292 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %293 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_63, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%293, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%293 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%294 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %296 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%295, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%296 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %289 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %300 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%299, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%300, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%300 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%301 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_64 = tensor.pad %302 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %303 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_64, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %304 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%303, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%303 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %306 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%305, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %307 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%306 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%307, %299 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %310 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%309, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %311 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%310 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%311 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_65 = tensor.pad %312 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %313 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_65, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%313, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%313 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%314 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %316 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%315, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %317 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%316 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %309 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%318 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %320 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%319, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%320 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %322 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_66 = tensor.pad %322 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %323 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_66, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%323 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %326 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%325, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %327 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%326, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%326 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %328 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%327, %319 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %329 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%328 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %330 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%329, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%330, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%330 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%331 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_67 = tensor.pad %332 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %333 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_67, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%333 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %336 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%335, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%336, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%336 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337, %329 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%338 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %340 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%339, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%340, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%340 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_68 = tensor.pad %342 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %343 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_68, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%343 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%344 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %346 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%345, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%346 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347, %339 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %350 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%349, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%350, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%350 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %352 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_69 = tensor.pad %352 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %353 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_69, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%353 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %356 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%355, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%356, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%356 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %349 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %360 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%359, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%360 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_70 = tensor.pad %362 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %363 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_70, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%363 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %366 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%365, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %367 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%366, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%366 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %359 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %370 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%369, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%370 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %372 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_71 = tensor.pad %372 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %373 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_71, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %374 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%373, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%373 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %376 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%375, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%376 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%377, %369 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %379 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%378 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %380 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%379, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%380, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%380 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_72 = tensor.pad %382 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %383 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_72, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%383, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%383 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%384 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %386 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%385, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%386 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %379 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %390 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%389, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%390, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%390 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%391 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_73 = tensor.pad %392 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %393 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_73, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%393 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %396 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%395, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %397 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%396, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%396 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%397, %389 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %399 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%398 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %400 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%399, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%400 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %402 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_74 = tensor.pad %402 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %403 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_74, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %404 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%403 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %405 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %406 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%405, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%406 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407, %399 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %410 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%409, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%410 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %412 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_75 = tensor.pad %412 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %413 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_75, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%413, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%413 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %416 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%415, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %417 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%416, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%416 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %409 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %420 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%419, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %421 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%420 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%421 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_76 = tensor.pad %422 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %423 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_76, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%423, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%423 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%424 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %426 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%425, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %427 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%426 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%427, %419 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%428 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %430 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%429, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%430, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%430 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_77 = tensor.pad %432 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %433 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_77, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%433 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%434 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %436 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%435, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%436 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%437, %429 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%438 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %440 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%439, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %441 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%440 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_78 = tensor.pad %442 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %443 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_78, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%443 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %446 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%445, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%446, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%446 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %439 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%448 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %450 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%449, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%450, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%450 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%451 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_79 = tensor.pad %452 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %453 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_79, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %454 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%453, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%453 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %456 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%455, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%456 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%457, %449 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %460 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%459, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %461 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%460 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%461 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_80 = tensor.pad %462 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %463 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_80, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%463 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %465 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %466 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%465, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%466, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%466 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467, %459 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %470 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%469, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%470 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_81 = tensor.pad %472 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %473 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_81, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%473 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %475 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %476 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%475, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %477 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%476 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%477, %469 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %479 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %480 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%479, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %481 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%480 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%481 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_82 = tensor.pad %482 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %483 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_82, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%483, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%483 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %486 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%485, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %487 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%486, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%486 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %479 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%488 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %490 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%489, %cst_26 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%490, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%490 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_83 = tensor.pad %492 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %493 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_83, %cst_21 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%138 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %494 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %cst_9, %cst_9, %cst_8, %cst_9 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%493 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x256x14x14xf32>
    %495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494 : tensor<1x256x14x14xf32>) outs(%137 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x256x14x14xf32>
    %496 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%495, %cst_22 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%143 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%496, %cst_24, %cst_24, %cst_23, %cst_24 : tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%496 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x1024x14x14xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497, %489 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1024x14x14xf32>
    %499 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%498 : tensor<1x1024x14x14xf32>) outs(%142 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x1024x14x14xf32>
    %500 = tensor.empty() : tensor<1x512x14x14xf32>
    %501 = linalg.fill ins(%cst_0 : f32) outs(%500 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %502 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%499, %cst_27 : tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) outs(%501 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%502, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x14x14xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%502 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x14x14xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503 : tensor<1x512x14x14xf32>) outs(%500 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_84 = tensor.pad %504 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %505 = tensor.empty() : tensor<1x512x7x7xf32>
    %506 = linalg.fill ins(%cst_0 : f32) outs(%505 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %507 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_84, %cst_28 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%506 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%507 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x7x7xf32>
    %509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508 : tensor<1x512x7x7xf32>) outs(%505 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x7x7xf32>
    %510 = tensor.empty() : tensor<1x2048x7x7xf32>
    %511 = linalg.fill ins(%cst_0 : f32) outs(%510 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %512 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%509, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%511 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %513 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%512, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%512 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x2048x7x7xf32>
    %514 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%499, %cst_32 : tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) outs(%511 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%514, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%514 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x2048x7x7xf32>
    %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%513, %515 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x2048x7x7xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516 : tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x2048x7x7xf32>
    %518 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%517, %cst_33 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%506 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %519 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%518, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%518 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x7x7xf32>
    %520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519 : tensor<1x512x7x7xf32>) outs(%505 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_85 = tensor.pad %520 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %521 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_85, %cst_28 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%506 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %522 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%521 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x7x7xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%522 : tensor<1x512x7x7xf32>) outs(%505 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x7x7xf32>
    %524 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%523, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%511 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %525 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%524, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%524 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x2048x7x7xf32>
    %526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525, %517 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x2048x7x7xf32>
    %527 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%526 : tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x2048x7x7xf32>
    %528 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%527, %cst_33 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%506 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%528 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x7x7xf32>
    %530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%529 : tensor<1x512x7x7xf32>) outs(%505 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_86 = tensor.pad %530 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %531 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_86, %cst_28 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%506 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531, %cst_17, %cst_17, %cst_16, %cst_17 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%531 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x512x7x7xf32>
    %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%532 : tensor<1x512x7x7xf32>) outs(%505 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x512x7x7xf32>
    %534 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%533, %cst_29 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%511 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %cst_31, %cst_31, %cst_30, %cst_31 : tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%534 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %in_89: f32, %in_90: f32, %in_91: f32, %out: f32):
      %550 = arith.truncf %cst_2 : f64 to f32
      %551 = arith.addf %in_91, %550 : f32
      %552 = math.rsqrt %551 : f32
      %553 = arith.subf %in, %in_90 : f32
      %554 = arith.mulf %553, %552 : f32
      %555 = arith.mulf %554, %in_88 : f32
      %556 = arith.addf %555, %in_89 : f32
      linalg.yield %556 : f32
    } -> tensor<1x2048x7x7xf32>
    %536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %527 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x2048x7x7xf32>
    %537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%536 : tensor<1x2048x7x7xf32>) outs(%510 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.cmpf ugt, %in, %cst_0 : f32
      %551 = arith.select %550, %in, %cst_0 : f32
      linalg.yield %551 : f32
    } -> tensor<1x2048x7x7xf32>
    %538 = tensor.empty() : tensor<7x7xi1>
    %539 = tensor.empty() : tensor<1x1xf32>
    %540 = tensor.empty() : tensor<1x2048x1x1xf32>
    %541 = linalg.fill ins(%cst_0 : f32) outs(%540 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %padded_87 = tensor.pad %537 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x2048x7x7xf32> to tensor<1x2048x8x8xf32>
    %542:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%538 : tensor<7x7xi1>) outs(%541, %539 : tensor<1x2048x1x1xf32>, tensor<1x1xf32>) {
    ^bb0(%in: i1, %out: f32, %out_88: f32):
      %550 = linalg.index 0 : index
      %551 = linalg.index 1 : index
      %552 = linalg.index 2 : index
      %553 = linalg.index 3 : index
      %554 = linalg.index 4 : index
      %555 = linalg.index 5 : index
      %556 = arith.muli %552, %c7 : index
      %557 = arith.addi %552, %c1 : index
      %558 = arith.muli %557, %c7 : index
      %559 = arith.muli %553, %c7 : index
      %560 = arith.addi %553, %c1 : index
      %561 = arith.muli %560, %c7 : index
      %562 = arith.addi %556, %554 : index
      %563 = arith.addi %559, %555 : index
      %extracted = tensor.extract %padded_87[%550, %551, %562, %563] : tensor<1x2048x8x8xf32>
      %564 = arith.cmpi ult, %562, %558 : index
      %565 = arith.select %564, %extracted, %cst_0 : f32
      %566 = arith.cmpi ult, %563, %561 : index
      %567 = arith.select %566, %565, %cst_0 : f32
      %568 = arith.addf %567, %out : f32
      %569 = arith.subi %558, %556 : index
      %570 = arith.subi %561, %559 : index
      %571 = arith.muli %569, %570 : index
      %572 = arith.index_cast %571 : index to i64
      %573 = arith.sitofp %572 : i64 to f32
      linalg.yield %568, %573 : f32, f32
    } -> (tensor<1x2048x1x1xf32>, tensor<1x1xf32>)
    %543 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%542#1 : tensor<1x1xf32>) outs(%542#0 : tensor<1x2048x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %550 = arith.divf %out, %in : f32
      linalg.yield %550 : f32
    } -> tensor<1x2048x1x1xf32>
    %collapsed = tensor.collapse_shape %543 [[0], [1, 2, 3]] : tensor<1x2048x1x1xf32> into tensor<1x2048xf32>
    %544 = tensor.empty() : tensor<2048x1000xf32>
    %545 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_34 : tensor<1000x2048xf32>) outs(%544 : tensor<2048x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2048x1000xf32>
    %546 = tensor.empty() : tensor<1x1000xf32>
    %547 = linalg.fill ins(%cst_0 : f32) outs(%546 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %548 = linalg.matmul ins(%collapsed, %545 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%547 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%548, %cst_35 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%546 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %550 = arith.addf %in, %in_88 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1000xf32>
    return %549 : tensor<1x1000xf32>
  }
}
