#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/socket.h>
#endif
#ifdef __linux__
#include <sys/socket.h>
#endif

#ifdef USE_IO_URING
#include <liburing.h>
#endif

#include <spdlog/spdlog.h>

namespace gossip_rl::rpc {

// Forward declarations
class EventLoop;
struct IoOperation;

// Completion callback type
using CompletionCallback = std::function<void(int result, void *user_data)>;

/**
 * Types of IO operations supported
 */
enum class IoOpType : uint8_t {
  READ,
  WRITE,
  ACCEPT,
  CONNECT,
  SEND,
  RECV,
  TIMEOUT,
  CANCEL,
  NOP
};

/**
 * IO Operation descriptor
 */
struct IoOperation {
  IoOpType type;
  int fd;
  void *buffer;
  size_t length;
  uint64_t offset;
  CompletionCallback callback;
  void *user_data;
  std::chrono::nanoseconds timeout{0};

  // For linked operations
  IoOperation *next{nullptr};
  bool is_linked{false};
};

/**
 * Event loop statistics for monitoring
 */
struct EventLoopStats {
  std::atomic<uint64_t> total_operations{0};
  std::atomic<uint64_t> completed_operations{0};
  std::atomic<uint64_t> failed_operations{0};
  std::atomic<uint64_t> timeouts{0};
  std::atomic<uint64_t> total_bytes_read{0};
  std::atomic<uint64_t> total_bytes_written{0};
  std::atomic<uint64_t> avg_completion_ns{0};

  void reset() {
    total_operations = 0;
    completed_operations = 0;
    failed_operations = 0;
    timeouts = 0;
    total_bytes_read = 0;
    total_bytes_written = 0;
    avg_completion_ns = 0;
  }
};

/**
 * High-performance event loop using io_uring on Linux
 * Falls back to kqueue on macOS or epoll as last resort
 */
class EventLoop {
public:
  struct Config {
    uint32_t ring_size = 4096;                 // SQ/CQ ring size
    uint32_t max_workers = 0;                  // 0 = auto-detect
    bool sqpoll = false;                       // Use SQ polling (requires root)
    bool iopoll = false;                       // Use IO polling for NVMe
    bool defer_taskrun = true;                 // Defer task running
    uint32_t batch_size = 32;                  // Submit batch size
    std::chrono::milliseconds idle_timeout{1}; // Idle wait timeout
  };

  explicit EventLoop(const Config &config);
  EventLoop(); // Default constructor
  ~EventLoop();

  // Non-copyable, non-movable
  EventLoop(const EventLoop &) = delete;
  EventLoop &operator=(const EventLoop &) = delete;
  EventLoop(EventLoop &&) = delete;
  EventLoop &operator=(EventLoop &&) = delete;

  /**
   * Start the event loop (blocks until stop() is called)
   */
  void run();

  /**
   * Run a single iteration of the event loop
   * @return Number of completions processed
   */
  int run_once();

  /**
   * Stop the event loop
   */
  void stop();

  /**
   * Check if the event loop is running
   */
  [[nodiscard]] bool is_running() const {
    return running_.load(std::memory_order_acquire);
  }

  /**
   * Submit a read operation
   */
  void submit_read(int fd, void *buffer, size_t length, uint64_t offset,
                   CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit a write operation
   */
  void submit_write(int fd, const void *buffer, size_t length, uint64_t offset,
                    CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit a socket send operation
   */
  void submit_send(int fd, const void *buffer, size_t length, int flags,
                   CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit a socket recv operation
   */
  void submit_recv(int fd, void *buffer, size_t length, int flags,
                   CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit an accept operation
   */
  void submit_accept(int listen_fd, sockaddr *addr, socklen_t *addrlen,
                     CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit a connect operation
   */
  void submit_connect(int fd, const sockaddr *addr, socklen_t addrlen,
                      CompletionCallback callback, void *user_data = nullptr);

  /**
   * Submit a timeout operation
   */
  void submit_timeout(std::chrono::nanoseconds duration,
                      CompletionCallback callback, void *user_data = nullptr);

  /**
   * Cancel a pending operation
   */
  void cancel(void *user_data);

  /**
   * Register buffers for zero-copy operations
   * @param buffers Vector of buffer spans to register
   * @return true on success
   */
  bool register_buffers(std::span<const std::span<uint8_t>> buffers);

  /**
   * Unregister previously registered buffers
   */
  void unregister_buffers();

  /**
   * Register file descriptors for faster access
   */
  bool register_files(std::span<const int> fds);

  /**
   * Get event loop statistics
   */
  [[nodiscard]] const EventLoopStats &stats() const { return stats_; }

  /**
   * Get the number of pending operations
   */
  [[nodiscard]] size_t pending_count() const {
    return pending_count_.load(std::memory_order_acquire);
  }

private:
  void process_completions();
  void submit_pending();

#ifdef USE_IO_URING
  struct io_uring ring_;
  bool ring_initialized_{false};
#endif

  Config config_;
  std::atomic<bool> running_{false};
  std::atomic<size_t> pending_count_{0};
  EventLoopStats stats_;

  // Pending operations queue
  std::vector<IoOperation> pending_ops_;
  std::mutex pending_mutex_;
};

/**
 * RAII guard for running event loop in a thread
 */
class EventLoopThread {
public:
  explicit EventLoopThread(std::shared_ptr<EventLoop> loop)
      : loop_(std::move(loop)) {
    thread_ = std::thread([this] { loop_->run(); });
  }

  ~EventLoopThread() {
    if (loop_) {
      loop_->stop();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  EventLoopThread(const EventLoopThread &) = delete;
  EventLoopThread &operator=(const EventLoopThread &) = delete;

  [[nodiscard]] EventLoop *get() const { return loop_.get(); }
  [[nodiscard]] EventLoop *operator->() const { return loop_.get(); }

private:
  std::shared_ptr<EventLoop> loop_;
  std::thread thread_;
};

} // namespace gossip_rl::rpc
