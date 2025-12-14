#include "rpc/buffer_pool.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gossip_rl::rpc {

BufferPool::BufferPool(const BufferPoolConfig &config) : config_(config) {
  allocate_buffers(config_.initial_count);
  spdlog::info("BufferPool initialized with {} buffers of {} bytes each",
               config_.initial_count, config_.buffer_size);
}

BufferPool::~BufferPool() {
  spdlog::info(
      "BufferPool destroyed. Stats: {} allocs, {} deallocs, peak usage: {}",
      stats_.allocations.load(), stats_.deallocations.load(),
      stats_.peak_usage.load());
}

void BufferPool::allocate_buffers(size_t count) {
  size_t current = buffer_storage_.size();

  // Allocate memory block
  size_t block_size = count * config_.buffer_size;
  auto block = std::make_unique<uint8_t[]>(block_size);
  uint8_t *block_ptr = block.get();

  // Create buffers pointing into the block
  buffer_storage_.reserve(current + count);
  for (size_t i = 0; i < count; ++i) {
    buffer_storage_.emplace_back(block_ptr + (i * config_.buffer_size),
                                 config_.buffer_size, current + i, false);
    free_list_.push_back(current + i);
  }

  memory_blocks_.push_back(std::move(block));
  total_buffers_.store(buffer_storage_.size(), std::memory_order_release);
}

bool BufferPool::expand_pool(size_t count) {
  if (total_buffers_.load() + count > config_.max_count) {
    count = config_.max_count - total_buffers_.load();
    if (count == 0) {
      return false;
    }
  }

  allocate_buffers(count);
  stats_.expansions.fetch_add(1, std::memory_order_relaxed);
  spdlog::debug("BufferPool expanded by {} buffers, total: {}", count,
                total_buffers_.load());
  return true;
}

std::unique_ptr<Buffer> BufferPool::acquire() {
  std::lock_guard lock(mutex_);

  if (free_list_.empty()) {
    // Try to expand
    if (!expand_pool(
            std::min(size_t{256}, config_.max_count - total_buffers_.load()))) {
      stats_.failed_allocations.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  }

  size_t index = free_list_.back();
  free_list_.pop_back();

  Buffer &stored = buffer_storage_[index];
  stored.reset();

  auto buffer =
      std::make_unique<Buffer>(stored.data(), stored.capacity(),
                               stored.pool_index(), stored.is_registered());

  stats_.allocations.fetch_add(1, std::memory_order_relaxed);
  uint64_t current =
      stats_.current_usage.fetch_add(1, std::memory_order_relaxed) + 1;
  uint64_t peak = stats_.peak_usage.load(std::memory_order_relaxed);
  while (current > peak && !stats_.peak_usage.compare_exchange_weak(
                               peak, current, std::memory_order_relaxed)) {
  }

  return buffer;
}

void BufferPool::release(std::unique_ptr<Buffer> buffer) {
  if (!buffer)
    return;

  std::lock_guard lock(mutex_);

  size_t index = buffer->pool_index();
  if (index >= buffer_storage_.size()) {
    spdlog::error("Invalid buffer pool index: {}", index);
    return;
  }

  free_list_.push_back(index);
  stats_.deallocations.fetch_add(1, std::memory_order_relaxed);
  stats_.current_usage.fetch_sub(1, std::memory_order_relaxed);
}

std::vector<std::span<uint8_t>> BufferPool::get_buffer_spans() const {
  std::lock_guard lock(mutex_);
  std::vector<std::span<uint8_t>> spans;
  spans.reserve(buffer_storage_.size());

  for (const auto &buf : buffer_storage_) {
    spans.emplace_back(const_cast<uint8_t *>(buf.data()), buf.capacity());
  }

  return spans;
}

void BufferPool::mark_registered() {
  registered_.store(true, std::memory_order_release);
}

// SharedMemoryRegion implementation

std::unique_ptr<SharedMemoryRegion>
SharedMemoryRegion::create(const std::string &name, size_t size) {

  std::string shm_name = "/" + name;

  // Remove existing if present
  shm_unlink(shm_name.c_str());

  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (fd < 0) {
    spdlog::error("Failed to create shared memory '{}': {}", name,
                  strerror(errno));
    return nullptr;
  }

  if (ftruncate(fd, size) < 0) {
    spdlog::error("Failed to set shared memory size: {}", strerror(errno));
    close(fd);
    shm_unlink(shm_name.c_str());
    return nullptr;
  }

  void *data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    spdlog::error("Failed to mmap shared memory: {}", strerror(errno));
    close(fd);
    shm_unlink(shm_name.c_str());
    return nullptr;
  }

  // Zero-initialize
  std::memset(data, 0, size);

  spdlog::info("Created shared memory region '{}' with {} bytes", name, size);
  return std::unique_ptr<SharedMemoryRegion>(
      new SharedMemoryRegion(name, fd, data, size));
}

std::unique_ptr<SharedMemoryRegion>
SharedMemoryRegion::open(const std::string &name) {
  std::string shm_name = "/" + name;

  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd < 0) {
    spdlog::error("Failed to open shared memory '{}': {}", name,
                  strerror(errno));
    return nullptr;
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    spdlog::error("Failed to stat shared memory: {}", strerror(errno));
    close(fd);
    return nullptr;
  }

  void *data =
      mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    spdlog::error("Failed to mmap shared memory: {}", strerror(errno));
    close(fd);
    return nullptr;
  }

  spdlog::info("Opened shared memory region '{}' with {} bytes", name,
               st.st_size);
  return std::unique_ptr<SharedMemoryRegion>(
      new SharedMemoryRegion(name, fd, data, st.st_size));
}

SharedMemoryRegion::~SharedMemoryRegion() {
  if (data_ && size_ > 0) {
    munmap(data_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
  if (!name_.empty()) {
    std::string shm_name = "/" + name_;
    shm_unlink(shm_name.c_str());
    spdlog::info("Destroyed shared memory region '{}'", name_);
  }
}

void SharedMemoryRegion::sync() {
  if (data_ && size_ > 0) {
    msync(data_, size_, MS_SYNC);
  }
}

} // namespace gossip_rl::rpc
