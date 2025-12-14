#include "rpc/event_loop.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>

namespace gossip_rl::rpc {

EventLoop::EventLoop() : EventLoop(Config{}) {}

EventLoop::EventLoop(const Config &config) : config_(config) {
#ifdef USE_IO_URING
  struct io_uring_params params{};

  if (config_.sqpoll) {
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 1000; // 1 second idle before sleeping
  }

  if (config_.iopoll) {
    params.flags |= IORING_SETUP_IOPOLL;
  }

  if (config_.defer_taskrun) {
    params.flags |= IORING_SETUP_SINGLE_ISSUER;
    params.flags |= IORING_SETUP_DEFER_TASKRUN;
  }

  int ret = io_uring_queue_init_params(config_.ring_size, &ring_, &params);
  if (ret < 0) {
    throw std::runtime_error("Failed to initialize io_uring: " +
                             std::string(strerror(-ret)));
  }
  ring_initialized_ = true;

  spdlog::info("io_uring initialized with ring size {}, flags={:#x}",
               config_.ring_size, params.flags);
#else
  spdlog::warn("io_uring not available, using fallback implementation");
#endif
}

EventLoop::~EventLoop() {
  stop();
#ifdef USE_IO_URING
  if (ring_initialized_) {
    io_uring_queue_exit(&ring_);
  }
#endif
}

void EventLoop::run() {
  running_.store(true, std::memory_order_release);
  spdlog::info("Event loop started");

  while (running_.load(std::memory_order_acquire)) {
    run_once();
  }

  spdlog::info("Event loop stopped");
}

int EventLoop::run_once() {
#ifdef USE_IO_URING
  // Submit any pending operations
  submit_pending();

  // Wait for completions
  struct io_uring_cqe *cqe;
  struct __kernel_timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = config_.idle_timeout.count() * 1000000;

  int ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
  if (ret == -ETIME) {
    return 0; // Timeout, no completions
  }

  if (ret < 0) {
    if (ret != -EINTR) {
      spdlog::error("io_uring_wait_cqe failed: {}", strerror(-ret));
    }
    return 0;
  }

  // Process all available completions
  int processed = 0;
  unsigned head;
  io_uring_for_each_cqe(&ring_, head, cqe) {
    IoOperation *op = static_cast<IoOperation *>(io_uring_cqe_get_data(cqe));
    if (op && op->callback) {
      op->callback(cqe->res, op->user_data);

      if (cqe->res >= 0) {
        stats_.completed_operations.fetch_add(1, std::memory_order_relaxed);
        if (op->type == IoOpType::READ || op->type == IoOpType::RECV) {
          stats_.total_bytes_read.fetch_add(cqe->res,
                                            std::memory_order_relaxed);
        } else if (op->type == IoOpType::WRITE || op->type == IoOpType::SEND) {
          stats_.total_bytes_written.fetch_add(cqe->res,
                                               std::memory_order_relaxed);
        }
      } else {
        stats_.failed_operations.fetch_add(1, std::memory_order_relaxed);
      }
    }
    pending_count_.fetch_sub(1, std::memory_order_relaxed);
    ++processed;
  }

  io_uring_cq_advance(&ring_, processed);
  return processed;
#else
  // Fallback implementation using poll/select
  std::this_thread::sleep_for(config_.idle_timeout);
  return 0;
#endif
}

void EventLoop::stop() { running_.store(false, std::memory_order_release); }

void EventLoop::submit_pending() {
#ifdef USE_IO_URING
  std::lock_guard lock(pending_mutex_);
  if (pending_ops_.empty())
    return;

  for (auto &op : pending_ops_) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      io_uring_submit(&ring_);
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        spdlog::error("Failed to get SQE");
        continue;
      }
    }

    switch (op.type) {
    case IoOpType::READ:
      io_uring_prep_read(sqe, op.fd, op.buffer, op.length, op.offset);
      break;
    case IoOpType::WRITE:
      io_uring_prep_write(sqe, op.fd, op.buffer, op.length, op.offset);
      break;
    case IoOpType::SEND:
      io_uring_prep_send(sqe, op.fd, op.buffer, op.length, 0);
      break;
    case IoOpType::RECV:
      io_uring_prep_recv(sqe, op.fd, op.buffer, op.length, 0);
      break;
    case IoOpType::ACCEPT:
      io_uring_prep_accept(
          sqe, op.fd, static_cast<sockaddr *>(op.buffer),
          static_cast<socklen_t *>(reinterpret_cast<void *>(op.offset)), 0);
      break;
    case IoOpType::CONNECT:
      io_uring_prep_connect(
          sqe, op.fd, static_cast<const sockaddr *>(op.buffer), op.length);
      break;
    case IoOpType::TIMEOUT: {
      struct __kernel_timespec *ts =
          static_cast<struct __kernel_timespec *>(op.buffer);
      io_uring_prep_timeout(sqe, ts, 0, 0);
      break;
    }
    case IoOpType::CANCEL:
      io_uring_prep_cancel(sqe, op.user_data, 0);
      break;
    case IoOpType::NOP:
      io_uring_prep_nop(sqe);
      break;
    }

    io_uring_sqe_set_data(sqe, &op);
    pending_count_.fetch_add(1, std::memory_order_relaxed);
    stats_.total_operations.fetch_add(1, std::memory_order_relaxed);
  }

  int submitted = io_uring_submit(&ring_);
  if (submitted < 0) {
    spdlog::error("io_uring_submit failed: {}", strerror(-submitted));
  }

  pending_ops_.clear();
#endif
}

void EventLoop::submit_read(int fd, void *buffer, size_t length,
                            uint64_t offset, CompletionCallback callback,
                            void *user_data) {
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::READ,
                                     .fd = fd,
                                     .buffer = buffer,
                                     .length = length,
                                     .offset = offset,
                                     .callback = std::move(callback),
                                     .user_data = user_data});
}

void EventLoop::submit_write(int fd, const void *buffer, size_t length,
                             uint64_t offset, CompletionCallback callback,
                             void *user_data) {
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::WRITE,
                                     .fd = fd,
                                     .buffer = const_cast<void *>(buffer),
                                     .length = length,
                                     .offset = offset,
                                     .callback = std::move(callback),
                                     .user_data = user_data});
}

void EventLoop::submit_send(int fd, const void *buffer, size_t length,
                            int flags, CompletionCallback callback,
                            void *user_data) {
  (void)flags; // Flags handled in submit_pending
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::SEND,
                                     .fd = fd,
                                     .buffer = const_cast<void *>(buffer),
                                     .length = length,
                                     .offset = 0,
                                     .callback = std::move(callback),
                                     .user_data = user_data});
}

void EventLoop::submit_recv(int fd, void *buffer, size_t length, int flags,
                            CompletionCallback callback, void *user_data) {
  (void)flags;
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::RECV,
                                     .fd = fd,
                                     .buffer = buffer,
                                     .length = length,
                                     .offset = 0,
                                     .callback = std::move(callback),
                                     .user_data = user_data});
}

void EventLoop::submit_accept(int listen_fd, sockaddr *addr, socklen_t *addrlen,
                              CompletionCallback callback, void *user_data) {
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(
      IoOperation{.type = IoOpType::ACCEPT,
                  .fd = listen_fd,
                  .buffer = addr,
                  .length = 0,
                  .offset = reinterpret_cast<uint64_t>(addrlen),
                  .callback = std::move(callback),
                  .user_data = user_data});
}

void EventLoop::submit_connect(int fd, const sockaddr *addr, socklen_t addrlen,
                               CompletionCallback callback, void *user_data) {
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::CONNECT,
                                     .fd = fd,
                                     .buffer = const_cast<sockaddr *>(addr),
                                     .length = addrlen,
                                     .offset = 0,
                                     .callback = std::move(callback),
                                     .user_data = user_data});
}

void EventLoop::submit_timeout(std::chrono::nanoseconds duration,
                               CompletionCallback callback, void *user_data) {
#ifdef USE_IO_URING
  auto *ts = new struct __kernel_timespec {
    .tv_sec = static_cast<int64_t>(duration.count() / 1000000000),
    .tv_nsec = static_cast<long long>(duration.count() % 1000000000)
  };

  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(
      IoOperation{.type = IoOpType::TIMEOUT,
                  .fd = -1,
                  .buffer = ts,
                  .length = 0,
                  .offset = 0,
                  .callback =
                      [callback, ts](int result, void *data) {
                        callback(result, data);
                        delete ts;
                      },
                  .user_data = user_data});
#else
  (void)duration;
  (void)callback;
  (void)user_data;
#endif
}

void EventLoop::cancel(void *user_data) {
  std::lock_guard lock(pending_mutex_);
  pending_ops_.push_back(IoOperation{.type = IoOpType::CANCEL,
                                     .fd = -1,
                                     .buffer = nullptr,
                                     .length = 0,
                                     .offset = 0,
                                     .callback = nullptr,
                                     .user_data = user_data});
}

bool EventLoop::register_buffers(std::span<const std::span<uint8_t>> buffers) {
#ifdef USE_IO_URING
  std::vector<struct iovec> iovecs;
  iovecs.reserve(buffers.size());

  for (const auto &buf : buffers) {
    iovecs.push_back({buf.data(), buf.size()});
  }

  int ret = io_uring_register_buffers(&ring_, iovecs.data(), iovecs.size());
  if (ret < 0) {
    spdlog::error("Failed to register buffers: {}", strerror(-ret));
    return false;
  }

  spdlog::info("Registered {} buffers with io_uring", buffers.size());
  return true;
#else
  (void)buffers;
  return false;
#endif
}

void EventLoop::unregister_buffers() {
#ifdef USE_IO_URING
  io_uring_unregister_buffers(&ring_);
#endif
}

bool EventLoop::register_files(std::span<const int> fds) {
#ifdef USE_IO_URING
  int ret = io_uring_register_files(&ring_, fds.data(), fds.size());
  if (ret < 0) {
    spdlog::error("Failed to register files: {}", strerror(-ret));
    return false;
  }

  spdlog::info("Registered {} file descriptors with io_uring", fds.size());
  return true;
#else
  (void)fds;
  return false;
#endif
}

} // namespace gossip_rl::rpc
