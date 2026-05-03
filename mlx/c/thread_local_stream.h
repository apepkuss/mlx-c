/* Copyright © 2023-2026 Apple Inc. */

#ifndef MLX_THREAD_LOCAL_STREAM_H
#define MLX_THREAD_LOCAL_STREAM_H

#include <stdbool.h>

#include "mlx/c/device.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_thread_local_stream ThreadLocalStream
 *
 * A thread-local stream wrapper. Each call to
 * `mlx_stream_from_thread_local_stream` from a given thread returns
 * (lazily creating if needed) a real per-thread `mlx_stream`, allowing
 * the same handle to be safely passed between threads while still
 * triggering the per-thread `CommandEncoder` registration that mlx
 * 0.31.2+ requires.
 */
/**@{*/

/**
 * Opaque thread-local stream handle.
 */
typedef struct mlx_thread_local_stream_ {
  void* ctx;
} mlx_thread_local_stream;

/**
 * Returns a new empty thread-local stream handle.
 */
mlx_thread_local_stream mlx_thread_local_stream_new(void);

/**
 * Allocate a new thread-local stream on `dev` and write it to `*res`.
 * The returned handle can be passed to other threads; resolving it via
 * `mlx_stream_from_thread_local_stream` yields a real `mlx_stream`
 * registered on the calling thread.
 */
int mlx_new_thread_local_stream(
    mlx_thread_local_stream* res,
    mlx_device dev);

/**
 * Resolve a thread-local stream into a real `mlx_stream` registered on
 * the calling thread. The first call from a new thread lazily creates
 * (and registers) the underlying stream; subsequent calls from the
 * same thread return the same stream.
 */
int mlx_stream_from_thread_local_stream(
    mlx_stream* res,
    mlx_thread_local_stream tls);

/**
 * Synchronize with the per-thread stream corresponding to `tls`.
 */
int mlx_synchronize_thread_local_stream(mlx_thread_local_stream tls);

/**
 * Copy-assign `src` to `*dst`.
 */
int mlx_thread_local_stream_set(
    mlx_thread_local_stream* dst,
    mlx_thread_local_stream src);

/**
 * Compare two thread-local stream handles for equality.
 */
bool mlx_thread_local_stream_equal(
    mlx_thread_local_stream lhs,
    mlx_thread_local_stream rhs);

/**
 * Get the device of a thread-local stream.
 */
int mlx_thread_local_stream_get_device(
    mlx_device* dev,
    mlx_thread_local_stream tls);

/**
 * Get the global index of a thread-local stream.
 */
int mlx_thread_local_stream_get_index(
    int* index,
    mlx_thread_local_stream tls);

/**
 * Free a thread-local stream handle.
 */
int mlx_thread_local_stream_free(mlx_thread_local_stream tls);

/**
 * Register every existing GPU stream's `CommandEncoder` on the
 * **calling thread**.
 *
 * mlx 0.31.2 made the metal backend's `CommandEncoder` thread-local,
 * which means a `Stream` created on thread A only has an encoder on
 * thread A. Arrays produced on thread A (e.g. model weights loaded
 * before the engine thread spawns) carry that stream; running them
 * through `mlx_eval` on thread B fails with
 * `"There is no Stream(gpu, X) in current thread."`.
 *
 * Calling this once at the entry of a worker thread (e.g. an
 * engine's run loop) walks `mlx::core::get_streams()` and invokes
 * `mlx::core::gpu::new_stream(s)` for each GPU stream — that call is
 * idempotent (`encoders.try_emplace`), so the existing global
 * `Stream` objects gain encoders on the worker thread without
 * creating new stream indices. Subsequent ops referencing those
 * streams from the worker thread eval correctly.
 */
int mlx_register_all_streams_on_current_thread(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
