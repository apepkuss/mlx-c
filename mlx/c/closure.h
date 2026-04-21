/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_CLOSURE_H
#define MLX_CLOSURE_H

#include "mlx/c/array.h"
#include "mlx/c/map.h"
#include "mlx/c/optional.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_closure Closures
 * MLX closure objects.
 */
/**@{*/

typedef struct mlx_closure_ {
  void* ctx;
} mlx_closure;
mlx_closure mlx_closure_new(void);
int mlx_closure_free(mlx_closure cls);
mlx_closure mlx_closure_new_func(
    int (*fun)(mlx_vector_array*, const mlx_vector_array));
mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_set(mlx_closure* cls, const mlx_closure src);
int mlx_closure_apply(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input);

/**
 * Zero-heap-allocation variant of `mlx_closure_apply` for hot-path callers
 * (e.g. per-layer compiled graph invocations inside a decode loop).
 *
 * Unlike `mlx_closure_apply`, this entry point never allocates the
 * `mlx_vector_array` wrapper struct on the heap. Inputs and outputs flow
 * through caller-provided flat `mlx_array` arrays. The caller is
 * responsible for:
 *   - passing `in_count` valid `mlx_array` handles in `ins`
 *   - pre-allocating `outs` with at least `out_capacity` freshly-created
 *     `mlx_array` handles (via `mlx_array_new()` — ctx may be NULL)
 *
 * On success:
 *   - `*out_count` is set to the number of outputs produced by the closure
 *   - `outs[0..*out_count]` are populated with the result arrays
 *   - `outs[*out_count..out_capacity]` are left untouched
 *
 * Returns non-zero if the output buffer is too small or the closure
 * itself reports an error.
 *
 * Motivation: `mlx_closure_apply` constructs/destructs two
 * `mlx_vector_array` heap objects per call, which shows up as a
 * measurable per-layer overhead in tight decode loops driven from
 * non-C++ language bindings (e.g. Rust, Swift). The upstream C++ API
 * and pybind bindings avoid this entirely by passing `std::vector`
 * by value/reference, which RVO/NRVO place on the caller's stack.
 */
int mlx_closure_apply_flat(
    mlx_array* outs,
    size_t* out_count,
    size_t out_capacity,
    mlx_closure cls,
    const mlx_array* ins,
    size_t in_count);

mlx_closure mlx_closure_new_unary(int (*fun)(mlx_array*, const mlx_array));

typedef struct mlx_closure_kwargs_ {
  void* ctx;
} mlx_closure_kwargs;
mlx_closure_kwargs mlx_closure_kwargs_new(void);
int mlx_closure_kwargs_free(mlx_closure_kwargs cls);
mlx_closure_kwargs mlx_closure_kwargs_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array));
mlx_closure_kwargs mlx_closure_kwargs_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_kwargs_set(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src);
int mlx_closure_kwargs_apply(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1);

typedef struct mlx_closure_value_and_grad_ {
  void* ctx;
} mlx_closure_value_and_grad;
mlx_closure_value_and_grad mlx_closure_value_and_grad_new(void);
int mlx_closure_value_and_grad_free(mlx_closure_value_and_grad cls);
mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array));
mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_value_and_grad_set(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src);
int mlx_closure_value_and_grad_apply(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input);

typedef struct mlx_closure_custom_ {
  void* ctx;
} mlx_closure_custom;
mlx_closure_custom mlx_closure_custom_new(void);
int mlx_closure_custom_free(mlx_closure_custom cls);
mlx_closure_custom mlx_closure_custom_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array));
mlx_closure_custom mlx_closure_custom_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_set(
    mlx_closure_custom* cls,
    const mlx_closure_custom src);
int mlx_closure_custom_apply(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2);

typedef struct mlx_closure_custom_jvp_ {
  void* ctx;
} mlx_closure_custom_jvp;
mlx_closure_custom_jvp mlx_closure_custom_jvp_new(void);
int mlx_closure_custom_jvp_free(mlx_closure_custom_jvp cls);
mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num));
mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_jvp_set(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src);
int mlx_closure_custom_jvp_apply(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num);

typedef struct mlx_closure_custom_vmap_ {
  void* ctx;
} mlx_closure_custom_vmap;
mlx_closure_custom_vmap mlx_closure_custom_vmap_new(void);
int mlx_closure_custom_vmap_free(mlx_closure_custom_vmap cls);
mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num));
mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_vmap_set(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src);
int mlx_closure_custom_vmap_apply(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
