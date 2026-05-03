/* Copyright © 2023-2026 Apple Inc. */

#include "mlx/c/thread_local_stream.h"

// `mlx::core::gpu::new_stream` is declared in the GPU backend header;
// public mlx aggregate headers don't pull this in.
#include "mlx/backend/gpu/eval.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/thread_local_stream.h"

extern "C" mlx_thread_local_stream mlx_thread_local_stream_new(void) {
  return mlx_thread_local_stream_new_();
}

extern "C" int mlx_new_thread_local_stream(
    mlx_thread_local_stream* res,
    mlx_device dev) {
  try {
    mlx_thread_local_stream_set_(
        *res,
        mlx::core::new_thread_local_stream(mlx_device_get_(dev)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_stream_from_thread_local_stream(
    mlx_stream* res,
    mlx_thread_local_stream tls) {
  try {
    mlx_stream_set_(
        *res,
        mlx::core::stream_from_thread_local_stream(
            mlx_thread_local_stream_get_(tls)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_synchronize_thread_local_stream(
    mlx_thread_local_stream tls) {
  try {
    mlx::core::synchronize(mlx_thread_local_stream_get_(tls));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_thread_local_stream_set(
    mlx_thread_local_stream* dst,
    mlx_thread_local_stream src) {
  try {
    mlx_thread_local_stream_set_(*dst, mlx_thread_local_stream_get_(src));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" bool mlx_thread_local_stream_equal(
    mlx_thread_local_stream lhs,
    mlx_thread_local_stream rhs) {
  try {
    return mlx_thread_local_stream_get_(lhs) ==
        mlx_thread_local_stream_get_(rhs);
  } catch (std::exception&) {
    return false;
  }
}

extern "C" int mlx_thread_local_stream_get_device(
    mlx_device* dev,
    mlx_thread_local_stream tls) {
  try {
    mlx_device_set_(*dev, mlx_thread_local_stream_get_(tls).device);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_thread_local_stream_get_index(
    int* index,
    mlx_thread_local_stream tls) {
  try {
    *index = mlx_thread_local_stream_get_(tls).index;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_thread_local_stream_free(mlx_thread_local_stream tls) {
  try {
    mlx_thread_local_stream_free_(tls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_register_all_streams_on_current_thread(void) {
  try {
    auto streams = mlx::core::get_streams();
    for (const auto& s : streams) {
      if (s.device == mlx::core::Device::gpu) {
        // gpu::new_stream uses encoders.try_emplace internally, so
        // re-registering an existing stream on this thread is a no-op
        // for streams already known and an emplace for streams that
        // were created on a different thread (which is exactly what we
        // want).
        mlx::core::gpu::new_stream(s);
      }
    }
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
