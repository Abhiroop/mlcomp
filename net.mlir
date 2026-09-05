#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "Net"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x1x32x32xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant dense<[-0.0206456669, -0.0601748675, 8.771870e-02, -0.02500958, 0.0864687934, 0.00141752011, -0.0470282957, 0.0554162636, -0.0306661334, -0.0381692871]> : tensor<10xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<10x84xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<84xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<84x120xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<120xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<120x400xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<16xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<16x6x5x5xf32>
    %cst_7 = arith.constant dense<[-0.119338326, 0.188050508, -0.197015792, -0.12665312, 0.172507912, -6.995130e-03]> : tensor<6xf32>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<6x1x5x5xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %cst_10 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x6x28x28xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_7 : tensor<6xf32>) outs(%0 : tensor<1x6x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x6x28x28xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_8 : tensor<1x1x32x32xf32>, tensor<6x1x5x5xf32>) outs(%1 : tensor<1x6x28x28xf32>) -> tensor<1x6x28x28xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x6x28x28xf32>) outs(%0 : tensor<1x6x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ugt, %in, %cst_9 : f32
      %36 = arith.select %35, %in, %cst_9 : f32
      linalg.yield %36 : f32
    } -> tensor<1x6x28x28xf32>
    %4 = tensor.empty() : tensor<1x6x14x14xf32>
    %5 = linalg.fill ins(%cst_10 : f32) outs(%4 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<1x6x28x28xf32>, tensor<2x2xf32>) outs(%5 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %8 = tensor.empty() : tensor<1x16x10x10xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<16xf32>) outs(%8 : tensor<1x16x10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x10x10xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%7, %cst_6 : tensor<1x6x14x14xf32>, tensor<16x6x5x5xf32>) outs(%9 : tensor<1x16x10x10xf32>) -> tensor<1x16x10x10xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x16x10x10xf32>) outs(%8 : tensor<1x16x10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ugt, %in, %cst_9 : f32
      %36 = arith.select %35, %in, %cst_9 : f32
      linalg.yield %36 : f32
    } -> tensor<1x16x10x10xf32>
    %12 = tensor.empty() : tensor<1x16x5x5xf32>
    %13 = linalg.fill ins(%cst_10 : f32) outs(%12 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
    %14 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%11, %6 : tensor<1x16x10x10xf32>, tensor<2x2xf32>) outs(%13 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
    %collapsed = tensor.collapse_shape %14 [[0], [1, 2, 3]] : tensor<1x16x5x5xf32> into tensor<1x400xf32>
    %15 = tensor.empty() : tensor<400x120xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<120x400xf32>) outs(%15 : tensor<400x120xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<400x120xf32>
    %17 = tensor.empty() : tensor<1x120xf32>
    %18 = linalg.fill ins(%cst_9 : f32) outs(%17 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %19 = linalg.matmul ins(%collapsed, %16 : tensor<1x400xf32>, tensor<400x120xf32>) outs(%18 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %20 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%19, %cst_3 : tensor<1x120xf32>, tensor<120xf32>) outs(%17 : tensor<1x120xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.addf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<1x120xf32>
    %21 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1x120xf32>) outs(%17 : tensor<1x120xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ugt, %in, %cst_9 : f32
      %36 = arith.select %35, %in, %cst_9 : f32
      linalg.yield %36 : f32
    } -> tensor<1x120xf32>
    %22 = tensor.empty() : tensor<120x84xf32>
    %23 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<84x120xf32>) outs(%22 : tensor<120x84xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<120x84xf32>
    %24 = tensor.empty() : tensor<1x84xf32>
    %25 = linalg.fill ins(%cst_9 : f32) outs(%24 : tensor<1x84xf32>) -> tensor<1x84xf32>
    %26 = linalg.matmul ins(%21, %23 : tensor<1x120xf32>, tensor<120x84xf32>) outs(%25 : tensor<1x84xf32>) -> tensor<1x84xf32>
    %27 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%26, %cst_1 : tensor<1x84xf32>, tensor<84xf32>) outs(%24 : tensor<1x84xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.addf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<1x84xf32>
    %28 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<1x84xf32>) outs(%24 : tensor<1x84xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ugt, %in, %cst_9 : f32
      %36 = arith.select %35, %in, %cst_9 : f32
      linalg.yield %36 : f32
    } -> tensor<1x84xf32>
    %29 = tensor.empty() : tensor<84x10xf32>
    %30 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<10x84xf32>) outs(%29 : tensor<84x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<84x10xf32>
    %31 = tensor.empty() : tensor<1x10xf32>
    %32 = linalg.fill ins(%cst_9 : f32) outs(%31 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %33 = linalg.matmul ins(%28, %30 : tensor<1x84xf32>, tensor<84x10xf32>) outs(%32 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %34 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%33, %cst : tensor<1x10xf32>, tensor<10xf32>) outs(%31 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.addf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<1x10xf32>
    return %34 : tensor<1x10xf32>
  }
}

