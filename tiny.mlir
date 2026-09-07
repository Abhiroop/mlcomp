#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "TinyModel"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant dense<[-0.257151306, -0.444812059]> : tensor<2xf32>
    %cst_0 = arith.constant dense<[[-0.466641545, 0.115759194, 0.138935328, -0.105179965], [-0.358416855, -0.238811493, 0.117786705, -0.0971438288]]> : tensor<2x4xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<2x4xf32>) outs(%0 : tensor<4x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x2xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1x4xf32>, tensor<4x2xf32>) outs(%3 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1x2xf32>, tensor<2xf32>) outs(%2 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.addf %in, %in_2 : f32
      linalg.yield %7 : f32
    } -> tensor<1x2xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x2xf32>) outs(%2 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.cmpf ugt, %in, %cst_1 : f32
      %8 = arith.select %7, %in, %cst_1 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    return %6 : tensor<1x2xf32>
  }
}

