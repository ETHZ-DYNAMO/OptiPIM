#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map4 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map9 = affine_map<() -> ()>
#map10 = affine_map<(d0, d1, d2, d3) -> ()>
#map11 = affine_map<(d0, d1) -> (d0, d1)>
#map12 = affine_map<(d0, d1) -> (d1, d0)>
#map13 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map16 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map20 = affine_map<(d0, d1) -> (0, d1)>
#map21 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "BertWrapper"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x128xi64>) -> (tensor<1x128x768xf32>, tensor<1x768xf32>) {
    %cst = arith.constant dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF0000000000000000010000000000000101000000000000020100000000000003010000000000000401000000000000050100000000000006010000000000000701000000000000080100000000000009010000000000000A010000000000000B010000000000000C010000000000000D010000000000000E010000000000000F0100000000000010010000000000001101000000000000120100000000000013010000000000001401000000000000150100000000000016010000000000001701000000000000180100000000000019010000000000001A010000000000001B010000000000001C010000000000001D010000000000001E010000000000001F0100000000000020010000000000002101000000000000220100000000000023010000000000002401000000000000250100000000000026010000000000002701000000000000280100000000000029010000000000002A010000000000002B010000000000002C010000000000002D010000000000002E010000000000002F0100000000000030010000000000003101000000000000320100000000000033010000000000003401000000000000350100000000000036010000000000003701000000000000380100000000000039010000000000003A010000000000003B010000000000003C010000000000003D010000000000003E010000000000003F0100000000000040010000000000004101000000000000420100000000000043010000000000004401000000000000450100000000000046010000000000004701000000000000480100000000000049010000000000004A010000000000004B010000000000004C010000000000004D010000000000004E010000000000004F0100000000000050010000000000005101000000000000520100000000000053010000000000005401000000000000550100000000000056010000000000005701000000000000580100000000000059010000000000005A010000000000005B010000000000005C010000000000005D010000000000005E010000000000005F0100000000000060010000000000006101000000000000620100000000000063010000000000006401000000000000650100000000000066010000000000006701000000000000680100000000000069010000000000006A010000000000006B010000000000006C010000000000006D010000000000006E010000000000006F0100000000000070010000000000007101000000000000720100000000000073010000000000007401000000000000750100000000000076010000000000007701000000000000780100000000000079010000000000007A010000000000007B010000000000007C010000000000007D010000000000007E010000000000007F0100000000000080010000000000008101000000000000820100000000000083010000000000008401000000000000850100000000000086010000000000008701000000000000880100000000000089010000000000008A010000000000008B010000000000008C010000000000008D010000000000008E010000000000008F0100000000000090010000000000009101000000000000920100000000000093010000000000009401000000000000950100000000000096010000000000009701000000000000980100000000000099010000000000009A010000000000009B010000000000009C010000000000009D010000000000009E010000000000009F01000000000000A001000000000000A101000000000000A201000000000000A301000000000000A401000000000000A501000000000000A601000000000000A701000000000000A801000000000000A901000000000000AA01000000000000AB01000000000000AC01000000000000AD01000000000000AE01000000000000AF01000000000000B001000000000000B101000000000000B201000000000000B301000000000000B401000000000000B501000000000000B601000000000000B701000000000000B801000000000000B901000000000000BA01000000000000BB01000000000000BC01000000000000BD01000000000000BE01000000000000BF01000000000000C001000000000000C101000000000000C201000000000000C301000000000000C401000000000000C501000000000000C601000000000000C701000000000000C801000000000000C901000000000000CA01000000000000CB01000000000000CC01000000000000CD01000000000000CE01000000000000CF01000000000000D001000000000000D101000000000000D201000000000000D301000000000000D401000000000000D501000000000000D601000000000000D701000000000000D801000000000000D901000000000000DA01000000000000DB01000000000000DC01000000000000DD01000000000000DE01000000000000DF01000000000000E001000000000000E101000000000000E201000000000000E301000000000000E401000000000000E501000000000000E601000000000000E701000000000000E801000000000000E901000000000000EA01000000000000EB01000000000000EC01000000000000ED01000000000000EE01000000000000EF01000000000000F001000000000000F101000000000000F201000000000000F301000000000000F401000000000000F501000000000000F601000000000000F701000000000000F801000000000000F901000000000000FA01000000000000FB01000000000000FC01000000000000FD01000000000000FE01000000000000FF01000000000000"> : tensor<1x512xi64>
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c30522 = arith.constant 30522 : index
    %c2 = arith.constant 2 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 9.9999999999999998E-13 : f64
    %cst_5 = arith.constant 7.680000e+02 : f32
    %cst_6 = arith.constant 1.41421354 : f32
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<30522x768xf32>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<2x768xf32>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<512x768xf32>
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<768x768xf32>
    %cst_12 = arith.constant dense<1.000000e+00> : tensor<3072xf32>
    %cst_13 = arith.constant dense<1.000000e+00> : tensor<3072x768xf32>
    %cst_14 = arith.constant dense<1.000000e+00> : tensor<768x3072xf32>
    %cst_15 = arith.constant dense<64> : tensor<i64>
    %cst_16 = arith.constant dense<0> : tensor<1x128xi64>
    %cst_17 = arith.constant dense<-3.4028234663852886E+38> : tensor<f64>
    %cst_18 = arith.constant dense<0xFFF0000000000000> : tensor<f64>
    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 128] [1, 1] : tensor<1x512xi64> to tensor<1x128xi64>
    %0 = tensor.empty() : tensor<1x128x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x128xi64>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %741 = arith.index_cast %in : i64 to index
      %742 = linalg.index 2 : index
      %743 = arith.cmpi slt, %741, %c30522 : index
      cf.assert %743, "index must be smaller than dim size"
      %744 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %744, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_7[%741, %742] : tensor<30522x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x128x768xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_16 : tensor<1x128xi64>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %741 = arith.index_cast %in : i64 to index
      %742 = linalg.index 2 : index
      %743 = arith.cmpi slt, %741, %c2 : index
      cf.assert %743, "index must be smaller than dim size"
      %744 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %744, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_8[%741, %742] : tensor<2x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x128x768xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %2 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x128xi64>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %741 = arith.index_cast %in : i64 to index
      %742 = linalg.index 2 : index
      %743 = arith.cmpi slt, %741, %c512 : index
      cf.assert %743, "index must be smaller than dim size"
      %744 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %744, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_9[%741, %742] : tensor<512x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x128x768xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %6 = tensor.empty() : tensor<1x128x1xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %9 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %10 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %10 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %11 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %14 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %16 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %17 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %21 = tensor.empty() : tensor<1x128xf32>
    %expanded = tensor.expand_shape %21 [[0], [1, 2, 3]] output_shape [1, 1, 1, 128] : tensor<1x128xf32> into tensor<1x1x1x128xf32>
    %22 = linalg.fill ins(%cst_1 : f32) outs(%expanded : tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %23 = tensor.empty() : tensor<1x1x128x128xf32>
    %24 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x1x1x128xf32>) outs(%23 : tensor<1x1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x128x128xf32>
    %25 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<1x1x128x128xf32>) outs(%23 : tensor<1x1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.subf %cst_1, %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x1x128x128xf32>
    %26 = tensor.empty() : tensor<1x1x128x128xi1>
    %27 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x1x128x128xf32>) outs(%26 : tensor<1x1x128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %741 = arith.cmpf une, %in, %cst_0 : f32
      linalg.yield %741 : i1
    } -> tensor<1x1x128x128xi1>
    %28 = tensor.empty() : tensor<f32>
    %29 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%cst_17 : tensor<f64>) outs(%28 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %741 = arith.truncf %in : f64 to f32
      linalg.yield %741 : f32
    } -> tensor<f32>
    %30 = linalg.generic {indexing_maps = [#map8, #map10, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %29, %25 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x1x128x128xf32>) outs(%23 : tensor<1x1x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x1x128x128xf32>
    %31 = tensor.empty() : tensor<768x768xf32>
    %32 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel"]} ins(%cst_11 : tensor<768x768xf32>) outs(%31 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %34 = tensor.empty() : tensor<1x768x768xf32>
    %35 = linalg.generic {indexing_maps = [#map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<768x768xf32>) outs(%34 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %36 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %37 = linalg.batch_matmul ins(%33, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_19 = tensor.expand_shape %38 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %39 = tensor.empty() : tensor<1x12x128x64xf32>
    %40 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_19 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %41 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%cst_15 : tensor<i64>) outs(%28 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %741 = arith.sitofp %in : i64 to f32
      linalg.yield %741 : f32
    } -> tensor<f32>
    %42 = tensor.empty() : tensor<1x12x64x128xf32>
    %43 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %44 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %45 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_20 = tensor.collapse_shape %45 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %46 = tensor.empty() : tensor<12x128x128xf32>
    %47 = linalg.fill ins(%cst_0 : f32) outs(%46 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %48 = linalg.batch_matmul ins(%collapsed, %collapsed_20 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_21 = tensor.expand_shape %48 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %49 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%41 : tensor<f32>) outs(%28 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.sqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<f32>
    %50 = tensor.empty() : tensor<1x12x128x128xf32>
    %51 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %52 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<1x1x128x128xf32>) outs(%26 : tensor<1x1x128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %741 = arith.cmpf oeq, %in, %cst_0 : f32
      linalg.yield %741 : i1
    } -> tensor<1x1x128x128xi1>
    %53 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%cst_18 : tensor<f64>) outs(%28 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %741 = arith.truncf %in : f64 to f32
      linalg.yield %741 : f32
    } -> tensor<f32>
    %54 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %51 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %55 = tensor.empty() : tensor<1x12x128xi64>
    %56 = linalg.fill ins(%c0_i64 : i64) outs(%55 : tensor<1x12x128xi64>) -> tensor<1x12x128xi64>
    %57 = tensor.empty() : tensor<1x12x128xf32>
    %58 = linalg.fill ins(%cst_2 : f32) outs(%57 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %59:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%54 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_22 = tensor.expand_shape %59#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %60 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54, %expanded_22 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %61 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %62 = tensor.empty() : tensor<1x12x128x1xf32>
    %63 = linalg.fill ins(%cst_0 : f32) outs(%62 : tensor<1x12x128x1xf32>) -> tensor<1x12x128x1xf32>
    %64 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%61 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %65 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %66 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_23 = tensor.collapse_shape %66 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %67 = tensor.empty() : tensor<12x128x64xf32>
    %68 = linalg.fill ins(%cst_0 : f32) outs(%67 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %69 = linalg.batch_matmul ins(%collapsed_23, %collapsed : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_24 = tensor.expand_shape %69 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %70 = tensor.empty() : tensor<1x128x12x64xf32>
    %71 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_25 = tensor.collapse_shape %71 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %72 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_25 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %73 = linalg.batch_matmul ins(%72, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %74 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %20 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %76 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %77 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %78 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %79 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %78 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %80 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %79 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %81 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %82 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %83 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %84 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %85 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %86 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %85 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %87 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %88 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %89 = tensor.empty() : tensor<768x3072xf32>
    %90 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<3072x768xf32>) outs(%89 : tensor<768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x3072xf32>
    %91 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %92 = tensor.empty() : tensor<1x768x3072xf32>
    %93 = linalg.generic {indexing_maps = [#map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90 : tensor<768x3072xf32>) outs(%92 : tensor<1x768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x3072xf32>
    %94 = tensor.empty() : tensor<1x128x3072xf32>
    %95 = linalg.fill ins(%cst_0 : f32) outs(%94 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %96 = linalg.batch_matmul ins(%91, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %97 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %98 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %99 = tensor.empty() : tensor<3072x768xf32>
    %100 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel"]} ins(%cst_14 : tensor<768x3072xf32>) outs(%99 : tensor<3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3072x768xf32>
    %101 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%98 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %102 = tensor.empty() : tensor<1x3072x768xf32>
    %103 = linalg.generic {indexing_maps = [#map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100 : tensor<3072x768xf32>) outs(%102 : tensor<1x3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3072x768xf32>
    %104 = linalg.batch_matmul ins(%101, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %105 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %106 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105, %88 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %107 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%106 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %108 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %109 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %110 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %109 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %111 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%110, %110 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %112 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %113 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %114 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %115 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %116 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %117 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%110, %116 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %118 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %119 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %120 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %121 = linalg.batch_matmul ins(%120, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %122 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%121, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_26 = tensor.expand_shape %122 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %123 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_26 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %124 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %125 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %126 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_27 = tensor.collapse_shape %125 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_28 = tensor.collapse_shape %126 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %127 = linalg.batch_matmul ins(%collapsed_27, %collapsed_28 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_29 = tensor.expand_shape %127 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %128 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_29, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %129 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %128 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %130:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%129 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_30 = tensor.expand_shape %130#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %131 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %expanded_30 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %132 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %133 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%132 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %134 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%132, %133 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %135 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_31 = tensor.collapse_shape %135 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %136 = linalg.batch_matmul ins(%collapsed_31, %collapsed_27 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_32 = tensor.expand_shape %136 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %137 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_32 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_33 = tensor.collapse_shape %137 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %138 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_33 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %139 = linalg.batch_matmul ins(%138, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %140 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%139, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %141 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %119 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %142 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%141 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %143 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %144 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%143 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %145 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%141, %144 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %146 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%145, %145 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %147 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%146 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %148 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%147 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %149 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%148 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %150 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%149 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %151 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %152 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%145, %151 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %153 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %154 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%153, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %155 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %156 = linalg.batch_matmul ins(%155, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %157 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %158 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %159 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %160 = linalg.batch_matmul ins(%159, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %161 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %162 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %154 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %163 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%162 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %164 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%163 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %165 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%164 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %166 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162, %165 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %167 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %166 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %168 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%167 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %169 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %170 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %171 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%170 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %172 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%171 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %173 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %172 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %174 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%173, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %175 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %176 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%175 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %177 = linalg.batch_matmul ins(%176, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %178 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%177, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_34 = tensor.expand_shape %178 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %179 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_34 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %180 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%179 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %181 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%179 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %182 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_35 = tensor.collapse_shape %181 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_36 = tensor.collapse_shape %182 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %183 = linalg.batch_matmul ins(%collapsed_35, %collapsed_36 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_37 = tensor.expand_shape %183 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %184 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_37, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %185 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %184 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %186:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%185 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_38 = tensor.expand_shape %186#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %187 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %expanded_38 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %188 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%187 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %189 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%188 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %190 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188, %189 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %191 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%190 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_39 = tensor.collapse_shape %191 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %192 = linalg.batch_matmul ins(%collapsed_39, %collapsed_35 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_40 = tensor.expand_shape %192 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %193 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_40 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_41 = tensor.collapse_shape %193 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %194 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_41 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %195 = linalg.batch_matmul ins(%194, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %196 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %197 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%196, %175 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %198 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%197 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %199 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %200 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%199 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %201 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197, %200 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %202 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201, %201 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %203 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%202 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %204 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%203 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %205 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %206 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%205 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %207 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%206 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %208 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201, %207 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %209 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%208, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %210 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%209, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %211 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%210 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %212 = linalg.batch_matmul ins(%211, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %213 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%212, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %214 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%213 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %215 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %216 = linalg.batch_matmul ins(%215, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %217 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%216, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %218 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%217, %210 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %219 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%218 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %220 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%219 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %221 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%220 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %222 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%218, %221 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %223 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %222 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %224 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%223 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %225 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%224 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %226 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%225 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %227 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%226 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %228 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %229 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %228 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %230 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%229, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %231 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%230, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %232 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %233 = linalg.batch_matmul ins(%232, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %234 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%233, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_42 = tensor.expand_shape %234 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %235 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_42 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %236 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %237 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %238 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%236 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_43 = tensor.collapse_shape %237 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_44 = tensor.collapse_shape %238 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %239 = linalg.batch_matmul ins(%collapsed_43, %collapsed_44 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_45 = tensor.expand_shape %239 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %240 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_45, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %241 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %240 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %242:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%241 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_46 = tensor.expand_shape %242#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %243 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %expanded_46 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %244 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %245 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%244 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %246 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %245 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %247 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%246 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_47 = tensor.collapse_shape %247 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %248 = linalg.batch_matmul ins(%collapsed_47, %collapsed_43 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_48 = tensor.expand_shape %248 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %249 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_48 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_49 = tensor.collapse_shape %249 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %250 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_49 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %251 = linalg.batch_matmul ins(%250, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %252 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%251, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %253 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252, %231 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %254 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%253 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %255 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %256 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%255 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %257 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%253, %256 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %258 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%257, %257 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %259 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%258 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %260 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %261 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%260 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %262 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%261 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %263 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%262 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %264 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%257, %263 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %265 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%264, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %266 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%265, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %267 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%266 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %268 = linalg.batch_matmul ins(%267, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %269 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%268, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %270 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%269 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %271 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%270 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %272 = linalg.batch_matmul ins(%271, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %273 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%272, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %274 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273, %266 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %275 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%274 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %276 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %277 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%276 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %278 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%274, %277 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %279 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278, %278 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %280 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%279 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %281 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%280 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %282 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %283 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%282 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %284 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%283 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %285 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278, %284 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %286 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%285, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %287 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%286, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %288 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %289 = linalg.batch_matmul ins(%288, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %290 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%289, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_50 = tensor.expand_shape %290 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %291 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_50 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %292 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %293 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %294 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%292 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_51 = tensor.collapse_shape %293 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_52 = tensor.collapse_shape %294 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %295 = linalg.batch_matmul ins(%collapsed_51, %collapsed_52 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_53 = tensor.expand_shape %295 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %296 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_53, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %297 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %296 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %298:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%297 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_54 = tensor.expand_shape %298#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %299 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %expanded_54 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %300 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %301 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%300 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %302 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%300, %301 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %303 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%302 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_55 = tensor.collapse_shape %303 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %304 = linalg.batch_matmul ins(%collapsed_55, %collapsed_51 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_56 = tensor.expand_shape %304 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %305 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_56 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_57 = tensor.collapse_shape %305 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %306 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_57 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %307 = linalg.batch_matmul ins(%306, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %308 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%307, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %309 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%308, %287 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%309 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %311 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%310 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %312 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%311 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %313 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%309, %312 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %314 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313, %313 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %315 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%314 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %316 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%315 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %317 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%316 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %318 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%317 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %319 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%318 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %320 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313, %319 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %321 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%320, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %322 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%321, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %323 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %324 = linalg.batch_matmul ins(%323, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %325 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%324, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %326 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%325 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %327 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%326 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %328 = linalg.batch_matmul ins(%327, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %329 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%328, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %330 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%329, %322 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %331 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%330 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %332 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%331 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %333 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%332 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %334 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%330, %333 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %335 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%334, %334 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %336 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%335 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %337 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%336 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %338 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %339 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%338 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %340 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%339 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %341 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%334, %340 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %342 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%341, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %343 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%342, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %344 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%343 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %345 = linalg.batch_matmul ins(%344, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %346 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%345, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_58 = tensor.expand_shape %346 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %347 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_58 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %348 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %349 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %350 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_59 = tensor.collapse_shape %349 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_60 = tensor.collapse_shape %350 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %351 = linalg.batch_matmul ins(%collapsed_59, %collapsed_60 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_61 = tensor.expand_shape %351 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %352 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_61, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %353 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %352 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %354:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%353 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_62 = tensor.expand_shape %354#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %355 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %expanded_62 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %356 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %357 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%356 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %358 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%356, %357 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %359 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_63 = tensor.collapse_shape %359 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %360 = linalg.batch_matmul ins(%collapsed_63, %collapsed_59 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_64 = tensor.expand_shape %360 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %361 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_64 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_65 = tensor.collapse_shape %361 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %362 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_65 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %363 = linalg.batch_matmul ins(%362, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %364 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%363, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %365 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%364, %343 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%365 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %367 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%366 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %368 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%367 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %369 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%365, %368 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %370 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369, %369 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %371 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%370 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %372 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%371 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %373 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%372 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %374 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%373 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %375 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%374 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %376 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369, %375 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %377 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%376, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %378 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%377, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %379 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%378 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %380 = linalg.batch_matmul ins(%379, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %381 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%380, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %382 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%381 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %383 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%382 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %384 = linalg.batch_matmul ins(%383, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %385 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%384, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %386 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%385, %378 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %387 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%386 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %388 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%387 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %389 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%388 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %390 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%386, %389 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %391 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%390, %390 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %392 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%391 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %393 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %394 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%393 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %395 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%394 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %396 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%395 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %397 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%390, %396 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %398 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%397, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %399 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%398, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %400 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%399 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %401 = linalg.batch_matmul ins(%400, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %402 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_66 = tensor.expand_shape %402 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %403 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_66 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %404 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %405 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %406 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_67 = tensor.collapse_shape %405 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_68 = tensor.collapse_shape %406 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %407 = linalg.batch_matmul ins(%collapsed_67, %collapsed_68 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_69 = tensor.expand_shape %407 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %408 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_69, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %409 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %408 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %410:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%409 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_70 = tensor.expand_shape %410#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %411 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %expanded_70 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %412 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %413 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%412 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %414 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %413 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %415 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_71 = tensor.collapse_shape %415 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %416 = linalg.batch_matmul ins(%collapsed_71, %collapsed_67 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_72 = tensor.expand_shape %416 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %417 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_72 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_73 = tensor.collapse_shape %417 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %418 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_73 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %419 = linalg.batch_matmul ins(%418, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %420 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%419, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %421 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%420, %399 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%421 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %423 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%422 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %424 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%423 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %425 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%421, %424 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %426 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425, %425 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %427 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%426 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %428 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %429 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%428 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %430 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%429 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %431 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%430 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %432 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425, %431 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %433 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%432, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %434 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%433, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %435 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%434 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %436 = linalg.batch_matmul ins(%435, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %437 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%436, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %438 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%437 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %439 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%438 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %440 = linalg.batch_matmul ins(%439, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %441 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%440, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %442 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%441, %434 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %443 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%442 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %444 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%443 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %445 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%444 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %446 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%442, %445 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %447 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%446, %446 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %448 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%447 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %449 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%448 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %450 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%449 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %451 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %452 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %453 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%446, %452 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %454 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%453, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %455 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%454, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %456 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%455 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %457 = linalg.batch_matmul ins(%456, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %458 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%457, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_74 = tensor.expand_shape %458 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %459 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_74 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %460 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %461 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %462 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_75 = tensor.collapse_shape %461 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_76 = tensor.collapse_shape %462 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %463 = linalg.batch_matmul ins(%collapsed_75, %collapsed_76 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_77 = tensor.expand_shape %463 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %464 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_77, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %465 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %464 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %466:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%465 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_78 = tensor.expand_shape %466#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %467 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%465, %expanded_78 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %468 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %469 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%468 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %470 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %469 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %471 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_79 = tensor.collapse_shape %471 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %472 = linalg.batch_matmul ins(%collapsed_79, %collapsed_75 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_80 = tensor.expand_shape %472 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %473 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_80 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_81 = tensor.collapse_shape %473 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %474 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_81 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %475 = linalg.batch_matmul ins(%474, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %476 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%475, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %477 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%476, %455 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %478 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%477 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %479 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%478 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %480 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %481 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%477, %480 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %482 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481, %481 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %483 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%482 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %484 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%483 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %485 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%484 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %486 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%485 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %487 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%486 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %488 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481, %487 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %489 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%488, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %490 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%489, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %491 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%490 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %492 = linalg.batch_matmul ins(%491, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %493 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%492, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %494 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%493 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %495 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%494 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %496 = linalg.batch_matmul ins(%495, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %497 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%496, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %498 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%497, %490 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %499 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%498 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %500 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%499 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %501 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%500 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %502 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%498, %501 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %503 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%502, %502 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %504 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%503 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %505 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %506 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %507 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%506 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %508 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %509 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%502, %508 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %510 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %511 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%510, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %512 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%511 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %513 = linalg.batch_matmul ins(%512, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %514 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%513, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_82 = tensor.expand_shape %514 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %515 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_82 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %516 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%515 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %517 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%515 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %518 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_83 = tensor.collapse_shape %517 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_84 = tensor.collapse_shape %518 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %519 = linalg.batch_matmul ins(%collapsed_83, %collapsed_84 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_85 = tensor.expand_shape %519 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %520 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_85, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %521 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %520 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %522:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%521 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_86 = tensor.expand_shape %522#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %523 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %expanded_86 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %524 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%523 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %525 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%524 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %526 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%524, %525 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %527 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%526 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_87 = tensor.collapse_shape %527 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %528 = linalg.batch_matmul ins(%collapsed_87, %collapsed_83 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_88 = tensor.expand_shape %528 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %529 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_88 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_89 = tensor.collapse_shape %529 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %530 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_89 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %531 = linalg.batch_matmul ins(%530, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %532 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%531, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %533 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%532, %511 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %534 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%533 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %535 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %536 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%535 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %537 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%533, %536 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %538 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%537, %537 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %539 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%538 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %540 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%539 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %541 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%540 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %542 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %543 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %544 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%537, %543 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %545 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %546 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %547 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %548 = linalg.batch_matmul ins(%547, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %549 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%548, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %550 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%549 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %551 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%550 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %552 = linalg.batch_matmul ins(%551, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %553 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%552, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %554 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553, %546 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %555 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%554 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %556 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %557 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%556 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %558 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%554, %557 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %559 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%558, %558 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %560 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%559 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %561 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%560 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %562 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%561 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %563 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%562 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %564 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%563 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %565 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%558, %564 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %566 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%565, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %567 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %568 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%567 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %569 = linalg.batch_matmul ins(%568, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %570 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%569, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_90 = tensor.expand_shape %570 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %571 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_90 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %572 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%571 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %573 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%571 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %574 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%572 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_91 = tensor.collapse_shape %573 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_92 = tensor.collapse_shape %574 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %575 = linalg.batch_matmul ins(%collapsed_91, %collapsed_92 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_93 = tensor.expand_shape %575 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %576 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_93, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %577 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %576 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %578:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%577 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_94 = tensor.expand_shape %578#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %579 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %expanded_94 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %580 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%579 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %581 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%580 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %582 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%580, %581 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %583 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%582 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_95 = tensor.collapse_shape %583 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %584 = linalg.batch_matmul ins(%collapsed_95, %collapsed_91 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_96 = tensor.expand_shape %584 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %585 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_96 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_97 = tensor.collapse_shape %585 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %586 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_97 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %587 = linalg.batch_matmul ins(%586, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %588 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%587, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %589 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%588, %567 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %590 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%589 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %591 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%590 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %592 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%591 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %593 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%589, %592 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %594 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%593, %593 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %595 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%594 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %596 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%595 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %597 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%596 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %598 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %599 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%598 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %600 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%593, %599 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %601 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%600, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %602 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %603 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%602 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %604 = linalg.batch_matmul ins(%603, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %605 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%604, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %606 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%605 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %607 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %608 = linalg.batch_matmul ins(%607, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %609 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%608, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %610 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%609, %602 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %611 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%610 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %612 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%611 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %613 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%612 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %614 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%610, %613 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %615 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614, %614 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %616 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%615 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %617 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%616 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %618 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%617 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %619 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%618 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %620 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%619 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %621 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614, %620 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %622 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %623 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%622, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %624 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%623 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %625 = linalg.batch_matmul ins(%624, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %626 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%625, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_98 = tensor.expand_shape %626 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %627 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_98 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %628 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %629 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %630 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%628 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_99 = tensor.collapse_shape %629 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_100 = tensor.collapse_shape %630 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %631 = linalg.batch_matmul ins(%collapsed_99, %collapsed_100 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_101 = tensor.expand_shape %631 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %632 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_101, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %633 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %632 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %634:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%633 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_102 = tensor.expand_shape %634#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %635 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%633, %expanded_102 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %636 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%635 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %637 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%636 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %638 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%636, %637 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %639 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%638 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_103 = tensor.collapse_shape %639 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %640 = linalg.batch_matmul ins(%collapsed_103, %collapsed_99 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_104 = tensor.expand_shape %640 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %641 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_104 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_105 = tensor.collapse_shape %641 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %642 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_105 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %643 = linalg.batch_matmul ins(%642, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %644 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%643, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %645 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%644, %623 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %646 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%645 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %647 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %648 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%647 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %649 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%645, %648 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %650 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%649, %649 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %651 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%650 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %652 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%651 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %653 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%652 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %654 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%653 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %655 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %656 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%649, %655 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %657 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%656, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %658 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%657, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %659 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%658 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %660 = linalg.batch_matmul ins(%659, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %661 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %662 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%661 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %663 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%662 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %664 = linalg.batch_matmul ins(%663, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %665 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%664, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %666 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%665, %658 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %667 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%666 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %668 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %669 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%668 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %670 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%666, %669 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %671 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670, %670 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %672 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%671 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %673 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%672 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %674 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%673 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %675 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%674 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %676 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %677 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670, %676 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %678 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %679 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%678, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %680 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%679 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %681 = linalg.batch_matmul ins(%680, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %682 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%681, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %expanded_106 = tensor.expand_shape %682 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %683 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_106 : tensor<1x128x12x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %684 = linalg.generic {indexing_maps = [#map7, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%683 : tensor<1x12x128x64xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %685 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%683 : tensor<1x12x128x64xf32>) outs(%39 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %686 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%684 : tensor<1x12x64x128xf32>) outs(%42 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_107 = tensor.collapse_shape %685 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_108 = tensor.collapse_shape %686 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %687 = linalg.batch_matmul ins(%collapsed_107, %collapsed_108 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_109 = tensor.expand_shape %687 [[0, 1], [2], [3]] output_shape [1, 12, 128, 128] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %688 = linalg.generic {indexing_maps = [#map16, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_109, %49 : tensor<1x12x128x128xf32>, tensor<f32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %689 = linalg.generic {indexing_maps = [#map8, #map10, #map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %53, %688 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_116: f32, %in_117: f32, %out: f32):
      %741 = arith.select %in, %in_116, %in_117 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %690:2 = linalg.generic {indexing_maps = [#map7, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%689 : tensor<1x12x128x128xf32>) outs(%58, %56 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_116: i64):
      %741 = linalg.index 3 : index
      %742 = arith.index_cast %741 : index to i64
      %743 = arith.maximumf %in, %out : f32
      %744 = arith.cmpf ogt, %in, %out : f32
      %745 = arith.select %744, %742, %out_116 : i64
      linalg.yield %743, %745 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_110 = tensor.expand_shape %690#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %691 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%689, %expanded_110 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %692 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.exp %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %693 = linalg.generic {indexing_maps = [#map7, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%692 : tensor<1x12x128x128xf32>) outs(%63 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x1xf32>
    %694 = linalg.generic {indexing_maps = [#map16, #map18, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%692, %693 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.divf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x12x128x128xf32>
    %695 = linalg.generic {indexing_maps = [#map16, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%694 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %collapsed_111 = tensor.collapse_shape %695 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %696 = linalg.batch_matmul ins(%collapsed_111, %collapsed_107 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%68 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_112 = tensor.expand_shape %696 [[0, 1], [2], [3]] output_shape [1, 12, 128, 64] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %697 = linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_112 : tensor<1x12x128x64xf32>) outs(%70 : tensor<1x128x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x12x64xf32>
    %collapsed_113 = tensor.collapse_shape %697 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %698 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_113 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %699 = linalg.batch_matmul ins(%698, %35 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %700 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%699, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %701 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%700, %679 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %702 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%701 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %703 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%702 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %704 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%703 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %705 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%701, %704 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %706 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%705, %705 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %707 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%706 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %708 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %709 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%708 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %710 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%709 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %711 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%710 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %712 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%705, %711 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %713 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%712, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %714 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%713, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %715 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%714 : tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %716 = linalg.batch_matmul ins(%715, %93 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%95 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %717 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716, %cst_12 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x3072xf32>
    %718 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%717 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_6 : f32
      %742 = math.erf %741 : f32
      %743 = arith.addf %742, %cst_1 : f32
      %744 = arith.mulf %743, %cst_3 : f32
      %745 = arith.mulf %in, %744 : f32
      linalg.yield %745 : f32
    } -> tensor<1x128x3072xf32>
    %719 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%718 : tensor<1x128x3072xf32>) outs(%94 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %720 = linalg.batch_matmul ins(%719, %103 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%36 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %721 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%720, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %722 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721, %714 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %723 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%722 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %724 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%723 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %725 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%724 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %726 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722, %725 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.subf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %727 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726, %726 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %728 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%727 : tensor<1x128x768xf32>) outs(%7 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.addf %in, %out : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %729 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.divf %in, %cst_5 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %730 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%729 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = arith.truncf %cst_4 : f64 to f32
      %742 = arith.addf %in, %741 : f32
      linalg.yield %742 : f32
    } -> tensor<1x128x1xf32>
    %731 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%730 : tensor<1x128x1xf32>) outs(%6 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.rsqrt %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x1xf32>
    %732 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%731 : tensor<1x128x1xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %733 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726, %732 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %734 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%733, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.mulf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %735 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%734, %cst_10 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x128x768xf32>
    %extracted_slice_114 = tensor.extract_slice %735[0, 0, 0] [1, 1, 768] [1, 1, 1] : tensor<1x128x768xf32> to tensor<1x1x768xf32>
    %collapsed_115 = tensor.collapse_shape %extracted_slice_114 [[0, 1], [2]] : tensor<1x1x768xf32> into tensor<1x768xf32>
    %736 = tensor.empty() : tensor<1x768xf32>
    %737 = linalg.fill ins(%cst_0 : f32) outs(%736 : tensor<1x768xf32>) -> tensor<1x768xf32>
    %738 = linalg.matmul ins(%collapsed_115, %32 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%737 : tensor<1x768xf32>) -> tensor<1x768xf32>
    %739 = linalg.generic {indexing_maps = [#map20, #map21, #map11], iterator_types = ["parallel", "parallel"]} ins(%738, %cst_10 : tensor<1x768xf32>, tensor<768xf32>) outs(%736 : tensor<1x768xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %741 = arith.addf %in, %in_116 : f32
      linalg.yield %741 : f32
    } -> tensor<1x768xf32>
    %740 = linalg.generic {indexing_maps = [#map20, #map11], iterator_types = ["parallel", "parallel"]} ins(%739 : tensor<1x768xf32>) outs(%736 : tensor<1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %741 = math.tanh %in : f32
      linalg.yield %741 : f32
    } -> tensor<1x768xf32>
    return %735, %740 : tensor<1x128x768xf32>, tensor<1x768xf32>
  }
}
