#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

namespace gossip_rl::rpc {

/**
 * A zero-copy buffer for network operations
 * Supports both registered io_uring buffers and regular allocations
 */
class Buffer {
public:
  Buffer() = default;
  Buffer(uint8_t *data, size_t capacity, size_t pool_index, bool registered)
      : data_(data), capacity_(capacity), size_(0), pool_index_(pool_index),
        registered_(registered) {}

  // Move-only semantics
  Buffer(Buffer &&other) noexcept
      : data_(other.data_), capacity_(other.capacity_), size_(other.size_),
        pool_index_(other.pool_index_), registered_(other.registered_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
    other.size_ = 0;
  }

  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      data_ = other.data_;
      capacity_ = other.capacity_;
      size_ = other.size_;
      pool_index_ = other.pool_index_;
      registered_ = other.registered_;
      other.data_ = nullptr;
      other.capacity_ = 0;
      other.size_ = 0;
    }
    return *this;
  }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  [[nodiscard]] uint8_t *data() { return data_; }
  [[nodiscard]] const uint8_t *data() const { return data_; }
  [[nodiscard]] size_t capacity() const { return capacity_; }
  [[nodiscard]] size_t size() const { return size_; }
  [[nodiscard]] size_t pool_index() const { return pool_index_; }
  [[nodiscard]] bool is_registered() const { return registered_; }
  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] size_t remaining() const { return capacity_ - size_; }

  void set_size(size_t size) { size_ = std::min(size, capacity_); }
  void reset() { size_ = 0; }

  // Span accessors
  [[nodiscard]] std::span<uint8_t> writable_span() {
    return {data_ + size_, capacity_ - size_};
  }

  [[nodiscard]] std::span<const uint8_t> readable_span() const {
    return {data_, size_};
  }

  // Write data to buffer
  size_t write(const void *src, size_t len) {
    size_t to_write = std::min(len, remaining());
    std::memcpy(data_ + size_, src, to_write);
    size_ += to_write;
    return to_write;
  }

  // Read data from buffer
  size_t read(void *dst, size_t len, size_t offset = 0) const {
    if (offset >= size_)
      return 0;
    size_t to_read = std::min(len, size_ - offset);
    std::memcpy(dst, data_ + offset, to_read);
    return to_read;
  }

private:
  uint8_t *data_{nullptr};
  size_t capacity_{0};
  size_t size_{0};
  size_t pool_index_{0};
  bool registered_{false};
};

/**
 * Buffer pool configuration
 */
struct BufferPoolConfig {
  size_t buffer_size = 64 * 1024; // 64KB default
  size_t initial_count = 1024;    // Initial buffers
  size_t max_count = 16384;       // Maximum buffers
  bool use_huge_pages = false;    // Use huge pages if available
  bool numa_aware = true;         // NUMA-aware allocation
  int numa_node = -1;             // -1 = auto-detect
};

/**
 * Statistics for the buffer pool
 */
struct BufferPoolStats {
  std::atomic<uint64_t> allocations{0};
  std::atomic<uint64_t> deallocations{0};
  std::atomic<uint64_t> current_usage{0};
  std::atomic<uint64_t> peak_usage{0};
  std::atomic<uint64_t> failed_allocations{0};
  std::atomic<uint64_t> expansions{0};

  void reset() {
    allocations = 0;
    deallocations = 0;
    current_usage = 0;
    peak_usage = 0;
    failed_allocations = 0;
    expansions = 0;
  }
};

/**
 * Lock-free buffer pool for zero-copy operations
 * Pre-allocates buffers and supports io_uring buffer registration
 */
class BufferPool {
public:
  explicit BufferPool(const BufferPoolConfig &config = BufferPoolConfig{});
  ~BufferPool();

  // Non-copyable, non-movable
  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;
  BufferPool(BufferPool &&) = delete;
  BufferPool &operator=(BufferPool &&) = delete;

  /**
   * Acquire a buffer from the pool
   * @return Buffer or nullptr if pool is exhausted
   */
  [[nodiscard]] std::unique_ptr<Buffer> acquire();

  /**
   * Release a buffer back to the pool
   */
  void release(std::unique_ptr<Buffer> buffer);

  /**
   * Get buffer spans for io_uring registration
   */
  [[nodiscard]] std::vector<std::span<uint8_t>> get_buffer_spans() const;

  /**
   * Mark buffers as registered with io_uring
   */
  void mark_registered();

  /**
   * Get pool statistics
   */
  [[nodiscard]] const BufferPoolStats &stats() const { return stats_; }

  /**
   * Get configuration
   */
  [[nodiscard]] const BufferPoolConfig &config() const { return config_; }

  /**
   * Get current pool size
   */
  [[nodiscard]] size_t pool_size() const {
    return total_buffers_.load(std::memory_order_acquire);
  }

  /**
   * Get number of available buffers
   */
  [[nodiscard]] size_t available() const {
    std::lock_guard lock(mutex_);
    return free_list_.size();
  }

private:
  bool expand_pool(size_t count);
  void allocate_buffers(size_t count);

  BufferPoolConfig config_;
  BufferPoolStats stats_;

  // Memory storage
  std::vector<std::unique_ptr<uint8_t[]>> memory_blocks_;
  std::vector<Buffer> buffer_storage_;

  // Free list (indices into buffer_storage_)
  std::vector<size_t> free_list_;
  mutable std::mutex mutex_;

  std::atomic<size_t> total_buffers_{0};
  std::atomic<bool> registered_{false};
};

/**
 * Shared memory region for intra-node zero-copy communication
 */
class SharedMemoryRegion {
public:
  /**
   * Create a new shared memory region
   * @param name Unique name for the region
   * @param size Size in bytes
   */
  static std::unique_ptr<SharedMemoryRegion> create(const std::string &name,
                                                    size_t size);

  /**
   * Open an existing shared memory region
   * @param name Name of the region to open
   */
  static std::unique_ptr<SharedMemoryRegion> open(const std::string &name);

  ~SharedMemoryRegion();

  // Non-copyable
  SharedMemoryRegion(const SharedMemoryRegion &) = delete;
  SharedMemoryRegion &operator=(const SharedMemoryRegion &) = delete;

  [[nodiscard]] void *data() { return data_; }
  [[nodiscard]] const void *data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }
  [[nodiscard]] const std::string &name() const { return name_; }
  [[nodiscard]] int fd() const { return fd_; }

  /**
   * Sync changes to shared memory
   */
  void sync();

private:
  SharedMemoryRegion(std::string name, int fd, void *data, size_t size)
      : name_(std::move(name)), fd_(fd), data_(data), size_(size) {}

  std::string name_;
  int fd_{-1};
  void *data_{nullptr};
  size_t size_{0};
};

/**
 * Ring buffer in shared memory for lock-free communication
 */
template <typename T> class SharedRingBuffer {
public:
  struct Header {
    std::atomic<uint64_t> write_pos{0};
    std::atomic<uint64_t> read_pos{0};
    uint64_t capacity{0};
    uint64_t element_size{0};
  };

  SharedRingBuffer(SharedMemoryRegion *region, size_t capacity)
      : region_(region) {
    header_ = reinterpret_cast<Header *>(region->data());
    data_ = reinterpret_cast<T *>(static_cast<uint8_t *>(region->data()) +
                                  sizeof(Header));

    // Initialize header if first creator
    if (header_->capacity == 0) {
      header_->capacity = capacity;
      header_->element_size = sizeof(T);
    }
  }

  /**
   * Try to push an element
   * @return true if successful
   */
  bool try_push(const T &value) {
    uint64_t write = header_->write_pos.load(std::memory_order_relaxed);
    uint64_t read = header_->read_pos.load(std::memory_order_acquire);

    if (write - read >= header_->capacity) {
      return false; // Full
    }

    data_[write % header_->capacity] = value;
    header_->write_pos.store(write + 1, std::memory_order_release);
    return true;
  }

  /**
   * Try to pop an element
   * @return true if successful
   */
  bool try_pop(T &value) {
    uint64_t read = header_->read_pos.load(std::memory_order_relaxed);
    uint64_t write = header_->write_pos.load(std::memory_order_acquire);

    if (read >= write) {
      return false; // Empty
    }

    value = data_[read % header_->capacity];
    header_->read_pos.store(read + 1, std::memory_order_release);
    return true;
  }

  [[nodiscard]] size_t size() const {
    uint64_t write = header_->write_pos.load(std::memory_order_acquire);
    uint64_t read = header_->read_pos.load(std::memory_order_acquire);
    return static_cast<size_t>(write - read);
  }

  [[nodiscard]] bool empty() const { return size() == 0; }
  [[nodiscard]] bool full() const { return size() >= header_->capacity; }
  [[nodiscard]] size_t capacity() const { return header_->capacity; }

private:
  SharedMemoryRegion *region_;
  Header *header_{nullptr};
  T *data_{nullptr};
};

} // namespace gossip_rl::rpc
