// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304 compile-mode=simd' | FileCheck %s --implicit-check-not=arith.shrsi
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304 compile-mode=simd' --triton-to-structured | FileCheck %s --check-prefix=LOWERED
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1 compile-mode=simd' | FileCheck %s --check-prefix=DISABLED
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304' | FileCheck %s --check-prefix=DISABLED
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304 compile-mode=simd_simt_template' | FileCheck %s --check-prefix=DISABLED
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304 compile-mode=unstructured_in_simt' | FileCheck %s --check-prefix=DISABLED
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=98304 compile-mode=simt_only' | FileCheck %s --check-prefix=DISABLED
// DISABLED-NOT: tt.gather
// DISABLED-NOT: gather.optimised.load

// LOWERED: tt.gather
// The early Gather rewrite must remain valid through structured conversion.
// A looped, tensor-built (tt.expand_dims/tt.broadcast) gather.
// Even on A2/A3, default/template modes must not run the explicit-SIMD rule.

// CHECK: arith.select
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: arith.cmpi sge, {{.*}}, %[[ZERO]]
// CHECK: scf.if
// CHECK:   tt.load {{%[^,]+}}, {{%[^,]+}}, {{%[^ ]+}} {gather.optimised.load = "source"} : tensor<2x16x64x!tt.ptr<f32>>
// CHECK:   tt.gather {{%[0-9]+}}[{{%[0-9]+}}] {axis = 2 : i32} : (tensor<2x16x64xf32>, tensor<2x16x128xi32>) -> tensor<2x16x128xf32>
// CHECK:   arith.select
// CHECK:   scf.yield
// CHECK:   else
// CHECK:   tt.load {{%[0-9]+, %[0-9]+, %cst_[0-9]+}} {gather.optimised.load = "fallback"} : tensor<2x16x128x!tt.ptr<f32>>

module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  tt.func public @scalar_loop_load_nd(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %witness_sink: !tt.ptr<f32>)  attributes {noinline = false} {
    %cst = arith.constant dense<-7.000000e+00> : tensor<2x16x128xf32>
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<2x16x128xi32>
    %cst_1 = arith.constant dense<128> : tensor<2x16x1xi32>
    %cst_2 = arith.constant dense<64> : tensor<2x16x1xi32>
    %cst_3 = arith.constant dense<16> : tensor<2x1xi32>
    %c65536_i32 = arith.constant 65536 : i32
    %c1639_i32 = arith.constant 1639 : i32
    %row_begin = tt.get_program_id x : i32
    %row_begin_4 = arith.muli %row_begin, %c1639_i32 : i32
    %row_end = arith.addi %row_begin_4, %c1639_i32 : i32
    %row_end_5 = arith.minsi %row_end, %c65536_i32 : i32
    %in_offsets_base = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %in_offsets_base_6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %in_offsets_base_7 = tt.expand_dims %in_offsets_base_6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %in_offsets_base_8 = tt.broadcast %in_offsets_base_7 : tensor<1x16xi32> -> tensor<2x16xi32>
    %mask = tt.splat %row_end_5 : i32 -> tensor<2xi32>
    %idx_offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %idx_offsets_9 = tt.expand_dims %idx_offsets {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %idx_offsets_10 = tt.expand_dims %idx_offsets_9 {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %idx_offsets_11 = tt.broadcast %idx_offsets_10 : tensor<1x1x128xi32> -> tensor<2x16x128xi32>
    %idx = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<2x16x128x!tt.ptr<i32>>
    %out = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<2x16x128x!tt.ptr<f32>>
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<2x16x128x!tt.ptr<f32>>
    scf.for %rb = %c0_i32 to %c1639_i32 step %c2_i32  : i32 {
      %in_offsets_base_12 = arith.addi %row_begin_4, %rb : i32
      %in_offsets_base_13 = tt.splat %in_offsets_base_12 : i32 -> tensor<2xi32>
      %in_offsets_base_14 = arith.addi %in_offsets_base_13, %in_offsets_base : tensor<2xi32>
      %in_offsets_base_15 = tt.expand_dims %in_offsets_base_14 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
      %in_offsets_base_16 = arith.muli %in_offsets_base_15, %cst_3 : tensor<2x1xi32>
      %in_offsets_base_17 = tt.broadcast %in_offsets_base_16 : tensor<2x1xi32> -> tensor<2x16xi32>
      %in_offsets_base_18 = arith.addi %in_offsets_base_17, %in_offsets_base_8 : tensor<2x16xi32>
      %in_offsets_base_19 = tt.expand_dims %in_offsets_base_18 {axis = 2 : i32} : tensor<2x16xi32> -> tensor<2x16x1xi32>
      %in_offsets_base_20 = arith.muli %in_offsets_base_19, %cst_2 : tensor<2x16x1xi32>
      %mask_21 = arith.cmpi slt, %in_offsets_base_14, %mask : tensor<2xi32>
      %mask_22 = tt.expand_dims %mask_21 {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
      %idx_offsets_23 = arith.muli %in_offsets_base_19, %cst_1 : tensor<2x16x1xi32>
      %idx_offsets_24 = tt.broadcast %idx_offsets_23 : tensor<2x16x1xi32> -> tensor<2x16x128xi32>
      %idx_offsets_25 = arith.addi %idx_offsets_24, %idx_offsets_11 : tensor<2x16x128xi32>
      %mask_26 = tt.expand_dims %mask_22 {axis = 2 : i32} : tensor<2x1xi1> -> tensor<2x1x1xi1>
      %idx_27 = tt.addptr %idx, %idx_offsets_25 : tensor<2x16x128x!tt.ptr<i32>>, tensor<2x16x128xi32>
      %idx_28 = tt.broadcast %mask_26 : tensor<2x1x1xi1> -> tensor<2x16x128xi1>
      %idx_29 = tt.load %idx_27, %idx_28, %cst_0 : tensor<2x16x128x!tt.ptr<i32>>
      %in_offsets = tt.broadcast %in_offsets_base_20 : tensor<2x16x1xi32> -> tensor<2x16x128xi32>
      %in_offsets_30 = arith.addi %in_offsets, %idx_29 : tensor<2x16x128xi32>
      %out_31 = tt.addptr %out, %in_offsets_30 : tensor<2x16x128x!tt.ptr<f32>>, tensor<2x16x128xi32>
      %readable_cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %readable_cols_0 = tt.expand_dims %readable_cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %readable_cols_1 = tt.expand_dims %readable_cols_0 {axis = 0 : i32} : tensor<1x64xi32> -> tensor<1x1x64xi32>
      %readable_grid = tt.broadcast %readable_cols_1 : tensor<1x1x64xi32> -> tensor<2x16x64xi32>
      %readable_base = tt.broadcast %in_offsets_base_20 : tensor<2x16x1xi32> -> tensor<2x16x64xi32>
      %readable_offsets = arith.addi %readable_base, %readable_grid : tensor<2x16x64xi32>
      %readable_src = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<2x16x64x!tt.ptr<f32>>
      %readable_ptrs = tt.addptr %readable_src, %readable_offsets : tensor<2x16x64x!tt.ptr<f32>>, tensor<2x16x64xi32>
      %readable_mask = tt.broadcast %mask_26 : tensor<2x1x1xi1> -> tensor<2x16x64xi1>
      %readable_zero = arith.constant dense<0.0> : tensor<2x16x64xf32>
      %readable = tt.load %readable_ptrs, %readable_mask, %readable_zero : tensor<2x16x64x!tt.ptr<f32>>
      %out_32 = tt.load %out_31, %idx_28, %cst : tensor<2x16x128x!tt.ptr<f32>>
      %readable_sink = tt.splat %witness_sink : !tt.ptr<f32> -> tensor<2x16x64x!tt.ptr<f32>>
      %readable_out = tt.addptr %readable_sink, %readable_offsets : tensor<2x16x64x!tt.ptr<f32>>, tensor<2x16x64xi32>
      tt.store %readable_out, %readable, %readable_mask : tensor<2x16x64x!tt.ptr<f32>>
      %1 = tt.addptr %0, %idx_offsets_25 : tensor<2x16x128x!tt.ptr<f32>>, tensor<2x16x128xi32>
      tt.store %1, %out_32, %idx_28 : tensor<2x16x128x!tt.ptr<f32>>
    }
    tt.return
  }
}
