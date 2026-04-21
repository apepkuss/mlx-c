/* Copyright © 2026 Apple Inc. */
/*
 * Benchmark comparing the two closure-apply entry points in a realistic
 * decode-style hot path (matmul -> compiled swiglu -> matmul).
 *
 * Baseline: `mlx_closure_apply` — heap-allocates mlx_vector_array
 *           wrappers on each call.
 * Candidate: `mlx_closure_apply_flat` — caller-provided flat arrays,
 *           zero mlx_vector_array heap traffic.
 */

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mlx/c/mlx.h"

#define ITERS 1000

static double now_ms(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) {
    mach_timebase_info(&tb);
  }
  uint64_t t = mach_absolute_time();
  return (double)t * (double)tb.numer / (double)tb.denom / 1.0e6;
}

/* swiglu: sigmoid(g) * g * u. */
static int swiglu_fun(mlx_vector_array* out, const mlx_vector_array in) {
  mlx_stream s = mlx_default_gpu_stream_new();
  mlx_array g = mlx_array_new();
  mlx_array u = mlx_array_new();
  mlx_vector_array_get(&g, in, 0);
  mlx_vector_array_get(&u, in, 1);

  mlx_array sig = mlx_array_new();
  mlx_sigmoid(&sig, g, s);
  mlx_array silu = mlx_array_new();
  mlx_multiply(&silu, g, sig, s);
  mlx_array h = mlx_array_new();
  mlx_multiply(&h, silu, u, s);

  mlx_vector_array_set_value(out, h);

  mlx_array_free(g);
  mlx_array_free(u);
  mlx_array_free(sig);
  mlx_array_free(silu);
  mlx_array_free(h);
  mlx_stream_free(s);
  return 0;
}

static mlx_array make_bf16_full(
    const int* shape, size_t ndim, float v, mlx_stream s) {
  mlx_array vals = mlx_array_new_float(v);
  mlx_array f = mlx_array_new();
  mlx_full(&f, shape, ndim, vals, MLX_FLOAT32, s);
  mlx_array out = mlx_array_new();
  mlx_astype(&out, f, MLX_BFLOAT16, s);
  mlx_array_free(vals);
  mlx_array_free(f);
  return out;
}

static int axes01[2] = {1, 0};

static void run_eager(
    mlx_array x, mlx_array wg, mlx_array wu, mlx_array wd, mlx_stream s) {
  mlx_array g = mlx_array_new();
  mlx_array u = mlx_array_new();
  mlx_array wg_t = mlx_array_new();
  mlx_array wu_t = mlx_array_new();
  mlx_array wd_t = mlx_array_new();
  mlx_transpose_axes(&wg_t, wg, axes01, 2, s);
  mlx_transpose_axes(&wu_t, wu, axes01, 2, s);
  mlx_transpose_axes(&wd_t, wd, axes01, 2, s);
  mlx_matmul(&g, x, wg_t, s);
  mlx_matmul(&u, x, wu_t, s);
  mlx_array sig = mlx_array_new();
  mlx_sigmoid(&sig, g, s);
  mlx_array silu = mlx_array_new();
  mlx_multiply(&silu, g, sig, s);
  mlx_array h = mlx_array_new();
  mlx_multiply(&h, silu, u, s);
  mlx_array out = mlx_array_new();
  mlx_matmul(&out, h, wd_t, s);
  mlx_array_eval(out);
  mlx_array_free(g);
  mlx_array_free(u);
  mlx_array_free(wg_t);
  mlx_array_free(wu_t);
  mlx_array_free(wd_t);
  mlx_array_free(sig);
  mlx_array_free(silu);
  mlx_array_free(h);
  mlx_array_free(out);
}

static void run_baseline(
    mlx_array x, mlx_array wg, mlx_array wu, mlx_array wd,
    mlx_closure compiled, mlx_stream s) {
  mlx_array g = mlx_array_new();
  mlx_array u = mlx_array_new();
  mlx_array wg_t = mlx_array_new();
  mlx_array wu_t = mlx_array_new();
  mlx_array wd_t = mlx_array_new();
  mlx_transpose_axes(&wg_t, wg, axes01, 2, s);
  mlx_transpose_axes(&wu_t, wu, axes01, 2, s);
  mlx_transpose_axes(&wd_t, wd, axes01, 2, s);
  mlx_matmul(&g, x, wg_t, s);
  mlx_matmul(&u, x, wu_t, s);
  mlx_array ins[2] = {g, u};
  mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
  mlx_vector_array res = mlx_vector_array_new();
  mlx_closure_apply(&res, compiled, in);
  mlx_array h = mlx_array_new();
  mlx_vector_array_get(&h, res, 0);
  mlx_array out = mlx_array_new();
  mlx_matmul(&out, h, wd_t, s);
  mlx_array_eval(out);
  mlx_array_free(g);
  mlx_array_free(u);
  mlx_array_free(wg_t);
  mlx_array_free(wu_t);
  mlx_array_free(wd_t);
  mlx_array_free(h);
  mlx_array_free(out);
  mlx_vector_array_free(in);
  mlx_vector_array_free(res);
}

static void run_flat(
    mlx_array x, mlx_array wg, mlx_array wu, mlx_array wd,
    mlx_closure compiled, mlx_stream s) {
  mlx_array g = mlx_array_new();
  mlx_array u = mlx_array_new();
  mlx_array wg_t = mlx_array_new();
  mlx_array wu_t = mlx_array_new();
  mlx_array wd_t = mlx_array_new();
  mlx_transpose_axes(&wg_t, wg, axes01, 2, s);
  mlx_transpose_axes(&wu_t, wu, axes01, 2, s);
  mlx_transpose_axes(&wd_t, wd, axes01, 2, s);
  mlx_matmul(&g, x, wg_t, s);
  mlx_matmul(&u, x, wu_t, s);
  mlx_array h = mlx_array_new();
  mlx_array outs[1] = {h};
  mlx_array ins[2] = {g, u};
  size_t out_count = 0;
  mlx_closure_apply_flat(outs, &out_count, 1, compiled, ins, 2);
  h = outs[0];
  mlx_array out = mlx_array_new();
  mlx_matmul(&out, h, wd_t, s);
  mlx_array_eval(out);
  mlx_array_free(g);
  mlx_array_free(u);
  mlx_array_free(wg_t);
  mlx_array_free(wu_t);
  mlx_array_free(wd_t);
  mlx_array_free(h);
  mlx_array_free(out);
}

int main(void) {
  const int hidden = 2560;
  const int inter = 9728;
  int x_shape[3] = {1, 1, hidden};
  int wg_shape[2] = {inter, hidden};
  int wd_shape[2] = {hidden, inter};

  mlx_stream s = mlx_default_gpu_stream_new();

  mlx_array x = make_bf16_full(x_shape, 3, 0.01f, s);
  mlx_array wg = make_bf16_full(wg_shape, 2, 0.001f, s);
  mlx_array wu = make_bf16_full(wg_shape, 2, 0.001f, s);
  mlx_array wd = make_bf16_full(wd_shape, 2, 0.001f, s);
  mlx_array_eval(x);
  mlx_array_eval(wg);
  mlx_array_eval(wu);
  mlx_array_eval(wd);

  mlx_closure raw = mlx_closure_new_func(swiglu_fun);
  mlx_closure compiled = mlx_closure_new();
  if (mlx_compile(&compiled, raw, /*shapeless=*/true) != 0) {
    fprintf(stderr, "mlx_compile failed\n");
    return 1;
  }

  /* Warmup each path. */
  for (int i = 0; i < 50; ++i) run_eager(x, wg, wu, wd, s);
  for (int i = 0; i < 50; ++i) run_baseline(x, wg, wu, wd, compiled, s);
  for (int i = 0; i < 50; ++i) run_flat(x, wg, wu, wd, compiled, s);

  double t0, eager_ms, base_ms, flat_ms;

  t0 = now_ms();
  for (int i = 0; i < ITERS; ++i) run_eager(x, wg, wu, wd, s);
  eager_ms = now_ms() - t0;

  t0 = now_ms();
  for (int i = 0; i < ITERS; ++i) run_baseline(x, wg, wu, wd, compiled, s);
  base_ms = now_ms() - t0;

  t0 = now_ms();
  for (int i = 0; i < ITERS; ++i) run_flat(x, wg, wu, wd, compiled, s);
  flat_ms = now_ms() - t0;

  printf("mlx-c closure-apply MLP pipeline bench (hidden=%d inter=%d bf16 x %d):\n",
         hidden, inter, ITERS);
  printf("  eager:                   %7.1f ms total, %.3f ms/call\n",
         eager_ms, eager_ms / ITERS);
  printf("  compiled (legacy apply): %7.1f ms total, %.3f ms/call   (%.2fx eager)\n",
         base_ms, base_ms / ITERS, base_ms / eager_ms);
  printf("  compiled (apply_flat):   %7.1f ms total, %.3f ms/call   (%.2fx eager)\n",
         flat_ms, flat_ms / ITERS, flat_ms / eager_ms);
  printf("  flat savings vs legacy:  %.3f ms/call (%.1f%% reduction)\n",
         (base_ms - flat_ms) / ITERS,
         (base_ms - flat_ms) / base_ms * 100.0);

  mlx_array_free(x);
  mlx_array_free(wg);
  mlx_array_free(wu);
  mlx_array_free(wd);
  mlx_closure_free(raw);
  mlx_closure_free(compiled);
  mlx_stream_free(s);
  return 0;
}
