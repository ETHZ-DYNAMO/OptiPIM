module attributes {torch.debug_module_name = "AlexNet"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x3x11x11xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c6 = arith.constant 6 : index
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<192x64x5x5xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<192xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<384x192x3x3xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<384xf32>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<256x384x3x3xf32>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<256x256x3x3xf32>
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<4096x9216xf32>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<4096xf32>
    %cst_12 = arith.constant dense<1.000000e+00> : tensor<4096x4096xf32>
    %cst_13 = arith.constant dense<1.000000e+00> : tensor<1000x4096xf32>
    %cst_14 = arith.constant dense<1.000000e+00> : tensor<1000xf32>
    %padded = tensor.pad %arg0 low[0, 0, 2, 2] high[0, 0, 2, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x228x228xf32>
    %0 = tensor.empty() : tensor<1x64x55x55xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<64xf32>) outs(%0 : tensor<1x64x55x55xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x55x55xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<4> : vector<2xi64>} ins(%padded, %cst : tensor<1x3x228x228xf32>, tensor<64x3x11x11xf32>) outs(%1 : tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x64x55x55xf32>) outs(%0 : tensor<1x64x55x55xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x64x55x55xf32>
    %4 = tensor.empty() : tensor<1x64x27x27xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<1x64x27x27xf32>) -> tensor<1x64x27x27xf32>
    %6 = tensor.empty() : tensor<3x3xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<1x64x55x55xf32>, tensor<3x3xf32>) outs(%5 : tensor<1x64x27x27xf32>) -> tensor<1x64x27x27xf32>
    %padded_15 = tensor.pad %7 low[0, 0, 2, 2] high[0, 0, 2, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x27x27xf32> to tensor<1x64x31x31xf32>
    %8 = tensor.empty() : tensor<1x192x27x27xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_4 : tensor<192xf32>) outs(%8 : tensor<1x192x27x27xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x192x27x27xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_15, %cst_3 : tensor<1x64x31x31xf32>, tensor<192x64x5x5xf32>) outs(%9 : tensor<1x192x27x27xf32>) -> tensor<1x192x27x27xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x192x27x27xf32>) outs(%8 : tensor<1x192x27x27xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x192x27x27xf32>
    %12 = tensor.empty() : tensor<1x192x13x13xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<1x192x13x13xf32>) -> tensor<1x192x13x13xf32>
    %14 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%11, %6 : tensor<1x192x27x27xf32>, tensor<3x3xf32>) outs(%13 : tensor<1x192x13x13xf32>) -> tensor<1x192x13x13xf32>
    %padded_16 = tensor.pad %14 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x192x13x13xf32> to tensor<1x192x15x15xf32>
    %15 = tensor.empty() : tensor<1x384x13x13xf32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_6 : tensor<384xf32>) outs(%15 : tensor<1x384x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x384x13x13xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_16, %cst_5 : tensor<1x192x15x15xf32>, tensor<384x192x3x3xf32>) outs(%16 : tensor<1x384x13x13xf32>) -> tensor<1x384x13x13xf32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x384x13x13xf32>) outs(%15 : tensor<1x384x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x384x13x13xf32>
    %padded_17 = tensor.pad %18 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x384x13x13xf32> to tensor<1x384x15x15xf32>
    %19 = tensor.empty() : tensor<1x256x13x13xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_8 : tensor<256xf32>) outs(%19 : tensor<1x256x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x256x13x13xf32>
    %21 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_17, %cst_7 : tensor<1x384x15x15xf32>, tensor<256x384x3x3xf32>) outs(%20 : tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<1x256x13x13xf32>) outs(%19 : tensor<1x256x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x256x13x13xf32>
    %padded_18 = tensor.pad %22 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x13x13xf32> to tensor<1x256x15x15xf32>
    %23 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_18, %cst_9 : tensor<1x256x15x15xf32>, tensor<256x256x3x3xf32>) outs(%20 : tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32>
    %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x256x13x13xf32>) outs(%19 : tensor<1x256x13x13xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x256x13x13xf32>
    %25 = tensor.empty() : tensor<1x256x6x6xf32>
    %26 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<1x256x6x6xf32>) -> tensor<1x256x6x6xf32>
    %27 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%24, %6 : tensor<1x256x13x13xf32>, tensor<3x3xf32>) outs(%26 : tensor<1x256x6x6xf32>) -> tensor<1x256x6x6xf32>
    %28 = tensor.empty() : tensor<2x2xi1>
    %29 = tensor.empty() : tensor<6x6xf32>
    %30 = linalg.fill ins(%cst_0 : f32) outs(%25 : tensor<1x256x6x6xf32>) -> tensor<1x256x6x6xf32>
    %padded_19 = tensor.pad %27 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x6x6xf32> to tensor<1x256x7x7xf32>
    %31:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%28 : tensor<2x2xi1>) outs(%30, %29 : tensor<1x256x6x6xf32>, tensor<6x6xf32>) {
    ^bb0(%in: i1, %out: f32, %out_20: f32):
      %51 = linalg.index 0 : index
      %52 = linalg.index 1 : index
      %53 = linalg.index 2 : index
      %54 = linalg.index 3 : index
      %55 = linalg.index 4 : index
      %56 = linalg.index 5 : index
      %57 = arith.muli %53, %c6 : index
      %58 = arith.floordivsi %57, %c6 : index
      %59 = arith.addi %53, %c1 : index
      %60 = arith.muli %59, %c6 : index
      %61 = arith.subi %60, %c1 : index
      %62 = arith.floordivsi %61, %c6 : index
      %63 = arith.addi %62, %c1 : index
      %64 = arith.muli %54, %c6 : index
      %65 = arith.floordivsi %64, %c6 : index
      %66 = arith.addi %54, %c1 : index
      %67 = arith.muli %66, %c6 : index
      %68 = arith.subi %67, %c1 : index
      %69 = arith.floordivsi %68, %c6 : index
      %70 = arith.addi %69, %c1 : index
      %71 = arith.addi %58, %55 : index
      %72 = arith.addi %65, %56 : index
      %extracted = tensor.extract %padded_19[%51, %52, %71, %72] : tensor<1x256x7x7xf32>
      %73 = arith.cmpi ult, %71, %63 : index
      %74 = arith.select %73, %extracted, %cst_0 : f32
      %75 = arith.cmpi ult, %72, %70 : index
      %76 = arith.select %75, %74, %cst_0 : f32
      %77 = arith.addf %76, %out : f32
      %78 = arith.subi %63, %58 : index
      %79 = arith.subi %70, %65 : index
      %80 = arith.muli %78, %79 : index
      %81 = arith.index_cast %80 : index to i64
      %82 = arith.sitofp %81 : i64 to f32
      linalg.yield %77, %82 : f32, f32
    } -> (tensor<1x256x6x6xf32>, tensor<6x6xf32>)
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31#1 : tensor<6x6xf32>) outs(%31#0 : tensor<1x256x6x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.divf %out, %in : f32
      linalg.yield %51 : f32
    } -> tensor<1x256x6x6xf32>
    %collapsed = tensor.collapse_shape %32 [[0], [1, 2, 3]] : tensor<1x256x6x6xf32> into tensor<1x9216xf32>
    %33 = tensor.empty() : tensor<9216x4096xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_10 : tensor<4096x9216xf32>) outs(%33 : tensor<9216x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<9216x4096xf32>
    %35 = tensor.empty() : tensor<1x4096xf32>
    %36 = linalg.fill ins(%cst_0 : f32) outs(%35 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %37 = linalg.matmul ins(%collapsed, %34 : tensor<1x9216xf32>, tensor<9216x4096xf32>) outs(%36 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %cst_11 : tensor<1x4096xf32>, tensor<4096xf32>) outs(%35 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %51 = arith.addf %in, %in_20 : f32
      linalg.yield %51 : f32
    } -> tensor<1x4096xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<1x4096xf32>) outs(%35 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x4096xf32>
    %40 = tensor.empty() : tensor<4096x4096xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_12 : tensor<4096x4096xf32>) outs(%40 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %42 = linalg.matmul ins(%39, %41 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%36 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%42, %cst_11 : tensor<1x4096xf32>, tensor<4096xf32>) outs(%35 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %51 = arith.addf %in, %in_20 : f32
      linalg.yield %51 : f32
    } -> tensor<1x4096xf32>
    %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%43 : tensor<1x4096xf32>) outs(%35 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.cmpf ugt, %in, %cst_0 : f32
      %52 = arith.select %51, %in, %cst_0 : f32
      linalg.yield %52 : f32
    } -> tensor<1x4096xf32>
    %45 = tensor.empty() : tensor<4096x1000xf32>
    %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<1000x4096xf32>) outs(%45 : tensor<4096x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x1000xf32>
    %47 = tensor.empty() : tensor<1x1000xf32>
    %48 = linalg.fill ins(%cst_0 : f32) outs(%47 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %49 = linalg.matmul ins(%44, %46 : tensor<1x4096xf32>, tensor<4096x1000xf32>) outs(%48 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%49, %cst_14 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%47 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %51 = arith.addf %in, %in_20 : f32
      linalg.yield %51 : f32
    } -> tensor<1x1000xf32>
    return %50 : tensor<1x1000xf32>
  }
}
