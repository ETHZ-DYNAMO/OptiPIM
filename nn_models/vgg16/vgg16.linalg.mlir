module attributes {torch.debug_module_name = "VGG"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x3x3x3xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c7 = arith.constant 7 : index
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<64x64x3x3xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x64x3x3xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<128x128x3x3xf32>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<256x128x3x3xf32>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<256x256x3x3xf32>
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<512x256x3x3xf32>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_12 = arith.constant dense<1.000000e+00> : tensor<512x512x3x3xf32>
    %cst_13 = arith.constant dense<1.000000e+00> : tensor<4096x25088xf32>
    %cst_14 = arith.constant dense<1.000000e+00> : tensor<4096xf32>
    %cst_15 = arith.constant dense<1.000000e+00> : tensor<4096x4096xf32>
    %cst_16 = arith.constant dense<1.000000e+00> : tensor<1000x4096xf32>
    %cst_17 = arith.constant dense<1.000000e+00> : tensor<1000xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
    %0 = tensor.empty() : tensor<1x64x224x224xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<64xf32>) outs(%0 : tensor<1x64x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x224x224xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst : tensor<1x3x226x226xf32>, tensor<64x3x3x3xf32>) outs(%1 : tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x64x224x224xf32>) outs(%0 : tensor<1x64x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x64x224x224xf32>
    %padded_18 = tensor.pad %3 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x224x224xf32> to tensor<1x64x226x226xf32>
    %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_18, %cst_3 : tensor<1x64x226x226xf32>, tensor<64x64x3x3xf32>) outs(%1 : tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x64x224x224xf32>) outs(%0 : tensor<1x64x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x64x224x224xf32>
    %6 = tensor.empty() : tensor<1x64x112x112xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = tensor.empty() : tensor<2x2xf32>
    %9 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%5, %8 : tensor<1x64x224x224xf32>, tensor<2x2xf32>) outs(%7 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %padded_19 = tensor.pad %9 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %10 = tensor.empty() : tensor<1x128x112x112xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<128xf32>) outs(%10 : tensor<1x128x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x112x112xf32>
    %12 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_19, %cst_4 : tensor<1x64x114x114xf32>, tensor<128x64x3x3xf32>) outs(%11 : tensor<1x128x112x112xf32>) -> tensor<1x128x112x112xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x128x112x112xf32>) outs(%10 : tensor<1x128x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x128x112x112xf32>
    %padded_20 = tensor.pad %13 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x112x112xf32> to tensor<1x128x114x114xf32>
    %14 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_20, %cst_6 : tensor<1x128x114x114xf32>, tensor<128x128x3x3xf32>) outs(%11 : tensor<1x128x112x112xf32>) -> tensor<1x128x112x112xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x128x112x112xf32>) outs(%10 : tensor<1x128x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x128x112x112xf32>
    %16 = tensor.empty() : tensor<1x128x56x56xf32>
    %17 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %18 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%15, %8 : tensor<1x128x112x112xf32>, tensor<2x2xf32>) outs(%17 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %padded_21 = tensor.pad %18 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
    %19 = tensor.empty() : tensor<1x256x56x56xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_8 : tensor<256xf32>) outs(%19 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x256x56x56xf32>
    %21 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_21, %cst_7 : tensor<1x128x58x58xf32>, tensor<256x128x3x3xf32>) outs(%20 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<1x256x56x56xf32>) outs(%19 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x256x56x56xf32>
    %padded_22 = tensor.pad %22 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x56x56xf32> to tensor<1x256x58x58xf32>
    %23 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_22, %cst_9 : tensor<1x256x58x58xf32>, tensor<256x256x3x3xf32>) outs(%20 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x256x56x56xf32>) outs(%19 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x256x56x56xf32>
    %padded_23 = tensor.pad %24 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x56x56xf32> to tensor<1x256x58x58xf32>
    %25 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_23, %cst_9 : tensor<1x256x58x58xf32>, tensor<256x256x3x3xf32>) outs(%20 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x256x56x56xf32>) outs(%19 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x256x56x56xf32>
    %27 = tensor.empty() : tensor<1x256x28x28xf32>
    %28 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %29 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%26, %8 : tensor<1x256x56x56xf32>, tensor<2x2xf32>) outs(%28 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %padded_24 = tensor.pad %29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x28x28xf32> to tensor<1x256x30x30xf32>
    %30 = tensor.empty() : tensor<1x512x28x28xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_11 : tensor<512xf32>) outs(%30 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x512x28x28xf32>
    %32 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_24, %cst_10 : tensor<1x256x30x30xf32>, tensor<512x256x3x3xf32>) outs(%31 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x512x28x28xf32>) outs(%30 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x28x28xf32>
    %padded_25 = tensor.pad %33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x28x28xf32> to tensor<1x512x30x30xf32>
    %34 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_25, %cst_12 : tensor<1x512x30x30xf32>, tensor<512x512x3x3xf32>) outs(%31 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<1x512x28x28xf32>) outs(%30 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x28x28xf32>
    %padded_26 = tensor.pad %35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x28x28xf32> to tensor<1x512x30x30xf32>
    %36 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_26, %cst_12 : tensor<1x512x30x30xf32>, tensor<512x512x3x3xf32>) outs(%31 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x512x28x28xf32>) outs(%30 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x28x28xf32>
    %38 = tensor.empty() : tensor<1x512x14x14xf32>
    %39 = linalg.fill ins(%cst_1 : f32) outs(%38 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %40 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%37, %8 : tensor<1x512x28x28xf32>, tensor<2x2xf32>) outs(%39 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %padded_27 = tensor.pad %40 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_11 : tensor<512xf32>) outs(%38 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x512x14x14xf32>
    %42 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_27, %cst_12 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%41 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<1x512x14x14xf32>) outs(%38 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_28 = tensor.pad %43 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %44 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_28, %cst_12 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%41 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44 : tensor<1x512x14x14xf32>) outs(%38 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_29 = tensor.pad %45 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %46 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_29, %cst_12 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%41 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<1x512x14x14xf32>) outs(%38 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x512x14x14xf32>
    %48 = tensor.empty() : tensor<1x512x7x7xf32>
    %49 = linalg.fill ins(%cst_1 : f32) outs(%48 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %50 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%47, %8 : tensor<1x512x14x14xf32>, tensor<2x2xf32>) outs(%49 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %51 = tensor.empty() : tensor<2x2xi1>
    %52 = tensor.empty() : tensor<7x7xf32>
    %53 = linalg.fill ins(%cst_0 : f32) outs(%48 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %padded_30 = tensor.pad %50 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x8x8xf32>
    %54:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%51 : tensor<2x2xi1>) outs(%53, %52 : tensor<1x512x7x7xf32>, tensor<7x7xf32>) {
    ^bb0(%in: i1, %out: f32, %out_31: f32):
      %74 = linalg.index 0 : index
      %75 = linalg.index 1 : index
      %76 = linalg.index 2 : index
      %77 = linalg.index 3 : index
      %78 = linalg.index 4 : index
      %79 = linalg.index 5 : index
      %80 = arith.muli %76, %c7 : index
      %81 = arith.floordivsi %80, %c7 : index
      %82 = arith.addi %76, %c1 : index
      %83 = arith.muli %82, %c7 : index
      %84 = arith.subi %83, %c1 : index
      %85 = arith.floordivsi %84, %c7 : index
      %86 = arith.addi %85, %c1 : index
      %87 = arith.muli %77, %c7 : index
      %88 = arith.floordivsi %87, %c7 : index
      %89 = arith.addi %77, %c1 : index
      %90 = arith.muli %89, %c7 : index
      %91 = arith.subi %90, %c1 : index
      %92 = arith.floordivsi %91, %c7 : index
      %93 = arith.addi %92, %c1 : index
      %94 = arith.addi %81, %78 : index
      %95 = arith.addi %88, %79 : index
      %extracted = tensor.extract %padded_30[%74, %75, %94, %95] : tensor<1x512x8x8xf32>
      %96 = arith.cmpi ult, %94, %86 : index
      %97 = arith.select %96, %extracted, %cst_0 : f32
      %98 = arith.cmpi ult, %95, %93 : index
      %99 = arith.select %98, %97, %cst_0 : f32
      %100 = arith.addf %99, %out : f32
      %101 = arith.subi %86, %81 : index
      %102 = arith.subi %93, %88 : index
      %103 = arith.muli %101, %102 : index
      %104 = arith.index_cast %103 : index to i64
      %105 = arith.sitofp %104 : i64 to f32
      linalg.yield %100, %105 : f32, f32
    } -> (tensor<1x512x7x7xf32>, tensor<7x7xf32>)
    %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54#1 : tensor<7x7xf32>) outs(%54#0 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.divf %out, %in : f32
      linalg.yield %74 : f32
    } -> tensor<1x512x7x7xf32>
    %collapsed = tensor.collapse_shape %55 [[0], [1, 2, 3]] : tensor<1x512x7x7xf32> into tensor<1x25088xf32>
    %56 = tensor.empty() : tensor<25088x4096xf32>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<4096x25088xf32>) outs(%56 : tensor<25088x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<25088x4096xf32>
    %58 = tensor.empty() : tensor<1x4096xf32>
    %59 = linalg.fill ins(%cst_0 : f32) outs(%58 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %60 = linalg.matmul ins(%collapsed, %57 : tensor<1x25088xf32>, tensor<25088x4096xf32>) outs(%59 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60, %cst_14 : tensor<1x4096xf32>, tensor<4096xf32>) outs(%58 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %74 = arith.addf %in, %in_31 : f32
      linalg.yield %74 : f32
    } -> tensor<1x4096xf32>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61 : tensor<1x4096xf32>) outs(%58 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x4096xf32>
    %63 = tensor.empty() : tensor<4096x4096xf32>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_15 : tensor<4096x4096xf32>) outs(%63 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %65 = linalg.matmul ins(%62, %64 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%59 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %cst_14 : tensor<1x4096xf32>, tensor<4096xf32>) outs(%58 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %74 = arith.addf %in, %in_31 : f32
      linalg.yield %74 : f32
    } -> tensor<1x4096xf32>
    %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66 : tensor<1x4096xf32>) outs(%58 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %74 = arith.cmpf ugt, %in, %cst_0 : f32
      %75 = arith.select %74, %in, %cst_0 : f32
      linalg.yield %75 : f32
    } -> tensor<1x4096xf32>
    %68 = tensor.empty() : tensor<4096x1000xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_16 : tensor<1000x4096xf32>) outs(%68 : tensor<4096x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x1000xf32>
    %70 = tensor.empty() : tensor<1x1000xf32>
    %71 = linalg.fill ins(%cst_0 : f32) outs(%70 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %72 = linalg.matmul ins(%67, %69 : tensor<1x4096xf32>, tensor<4096x1000xf32>) outs(%71 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %cst_17 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%70 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %74 = arith.addf %in, %in_31 : f32
      linalg.yield %74 : f32
    } -> tensor<1x1000xf32>
    return %73 : tensor<1x1000xf32>
  }
}
