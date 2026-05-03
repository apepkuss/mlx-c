/* Copyright © 2023-2026 Apple Inc.                   */

#ifndef MLX_THREAD_LOCAL_STREAM_PRIVATE_H
#define MLX_THREAD_LOCAL_STREAM_PRIVATE_H

#include "mlx/c/thread_local_stream.h"
#include "mlx/mlx.h"

inline mlx_thread_local_stream mlx_thread_local_stream_new_() {
  return mlx_thread_local_stream({nullptr});
}

inline mlx_thread_local_stream mlx_thread_local_stream_new_(
    const mlx::core::ThreadLocalStream& s) {
  return mlx_thread_local_stream({new mlx::core::ThreadLocalStream(s)});
}

inline mlx_thread_local_stream mlx_thread_local_stream_new_(
    mlx::core::ThreadLocalStream&& s) {
  return mlx_thread_local_stream(
      {new mlx::core::ThreadLocalStream(std::move(s))});
}

inline mlx_thread_local_stream& mlx_thread_local_stream_set_(
    mlx_thread_local_stream& d,
    const mlx::core::ThreadLocalStream& s) {
  if (d.ctx) {
    *static_cast<mlx::core::ThreadLocalStream*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::ThreadLocalStream(s);
  }
  return d;
}

inline mlx::core::ThreadLocalStream& mlx_thread_local_stream_get_(
    mlx_thread_local_stream d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_thread_local_stream");
  }
  return *static_cast<mlx::core::ThreadLocalStream*>(d.ctx);
}

inline void mlx_thread_local_stream_free_(mlx_thread_local_stream d) {
  if (d.ctx) {
    delete static_cast<mlx::core::ThreadLocalStream*>(d.ctx);
  }
}

#endif
