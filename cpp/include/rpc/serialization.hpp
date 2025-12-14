#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "rpc/buffer_pool.hpp"

namespace gossip_rl::rpc {

/**
 * Zero-copy serialization utilities using FlatBuffers-like approach
 * This is a minimal implementation for gradient tensors and RPC messages
 */

/**
 * Tensor data types
 */
enum class TensorDtype : uint8_t {
  FLOAT32 = 0,
  FLOAT16 = 1,
  BFLOAT16 = 2,
  INT32 = 3,
  INT64 = 4,
  UINT8 = 5,
  INT8 = 6
};

/**
 * Get byte size of tensor dtype
 */
constexpr size_t dtype_size(TensorDtype dtype) {
  switch (dtype) {
  case TensorDtype::FLOAT32:
    return 4;
  case TensorDtype::FLOAT16:
    return 2;
  case TensorDtype::BFLOAT16:
    return 2;
  case TensorDtype::INT32:
    return 4;
  case TensorDtype::INT64:
    return 8;
  case TensorDtype::UINT8:
    return 1;
  case TensorDtype::INT8:
    return 1;
  default:
    return 0;
  }
}

/**
 * Tensor header for serialization
 */
struct alignas(8) TensorHeader {
  uint32_t magic = 0x54454E53; // "TENS"
  TensorDtype dtype;
  uint8_t ndim;
  uint16_t reserved;
  uint64_t numel; // Number of elements
  // Followed by: dims[ndim] as uint64_t, then data

  static constexpr size_t BASE_SIZE = 16;
};

/**
 * Gradient message header
 */
struct alignas(8) GradientHeader {
  uint32_t magic = 0x47524144; // "GRAD"
  uint32_t num_tensors;
  uint64_t step;
  uint64_t total_size;
  uint32_t compression_type; // 0=none, 1=topk, 2=random, 3=quantized
  float compression_ratio;
  // Followed by tensor offsets: uint64_t[num_tensors]

  static constexpr size_t BASE_SIZE = 32;
};

/**
 * Sparse tensor representation for TopK compression
 */
struct SparseTensorHeader {
  uint32_t magic = 0x53505253; // "SPRS"
  TensorDtype dtype;
  uint8_t ndim;
  uint16_t reserved;
  uint64_t original_numel;
  uint64_t nnz; // Number of non-zeros
  // Followed by: dims[ndim], indices[nnz], values[nnz]

  static constexpr size_t BASE_SIZE = 24;
};

/**
 * Zero-copy buffer writer
 */
class BufferWriter {
public:
  explicit BufferWriter(Buffer *buffer) : buffer_(buffer) {}

  /**
   * Write raw bytes
   */
  bool write(const void *data, size_t size) {
    if (buffer_->remaining() < size)
      return false;
    std::memcpy(buffer_->data() + buffer_->size(), data, size);
    buffer_->set_size(buffer_->size() + size);
    return true;
  }

  /**
   * Write a value
   */
  template <typename T> bool write_value(const T &value) {
    return write(&value, sizeof(T));
  }

  /**
   * Write a span
   */
  template <typename T> bool write_span(std::span<const T> data) {
    return write(data.data(), data.size() * sizeof(T));
  }

  /**
   * Write a string (length-prefixed)
   */
  bool write_string(const std::string &str) {
    if (!write_value(static_cast<uint32_t>(str.size())))
      return false;
    return write(str.data(), str.size());
  }

  /**
   * Reserve space and return pointer
   */
  uint8_t *reserve(size_t size) {
    if (buffer_->remaining() < size)
      return nullptr;
    uint8_t *ptr = buffer_->data() + buffer_->size();
    buffer_->set_size(buffer_->size() + size);
    return ptr;
  }

  /**
   * Get current position
   */
  [[nodiscard]] size_t position() const { return buffer_->size(); }

  /**
   * Get remaining capacity
   */
  [[nodiscard]] size_t remaining() const { return buffer_->remaining(); }

private:
  Buffer *buffer_;
};

/**
 * Zero-copy buffer reader
 */
class BufferReader {
public:
  BufferReader(const uint8_t *data, size_t size)
      : data_(data), size_(size), pos_(0) {}

  explicit BufferReader(const Buffer &buffer)
      : data_(buffer.data()), size_(buffer.size()), pos_(0) {}

  /**
   * Read raw bytes
   */
  bool read(void *dest, size_t size) {
    if (remaining() < size)
      return false;
    std::memcpy(dest, data_ + pos_, size);
    pos_ += size;
    return true;
  }

  /**
   * Read a value
   */
  template <typename T> bool read_value(T &value) {
    return read(&value, sizeof(T));
  }

  /**
   * Peek at data without consuming
   */
  template <typename T> const T *peek() const {
    if (remaining() < sizeof(T))
      return nullptr;
    return reinterpret_cast<const T *>(data_ + pos_);
  }

  /**
   * Get span of remaining data
   */
  [[nodiscard]] std::span<const uint8_t> remaining_span() const {
    return {data_ + pos_, size_ - pos_};
  }

  /**
   * Read a string (length-prefixed)
   */
  bool read_string(std::string &str) {
    uint32_t len;
    if (!read_value(len))
      return false;
    if (remaining() < len)
      return false;
    str.assign(reinterpret_cast<const char *>(data_ + pos_), len);
    pos_ += len;
    return true;
  }

  /**
   * Skip bytes
   */
  bool skip(size_t bytes) {
    if (remaining() < bytes)
      return false;
    pos_ += bytes;
    return true;
  }

  /**
   * Get current position
   */
  [[nodiscard]] size_t position() const { return pos_; }

  /**
   * Get remaining bytes
   */
  [[nodiscard]] size_t remaining() const { return size_ - pos_; }

  /**
   * Reset to beginning
   */
  void reset() { pos_ = 0; }

  /**
   * Seek to position
   */
  bool seek(size_t pos) {
    if (pos > size_)
      return false;
    pos_ = pos;
    return true;
  }

private:
  const uint8_t *data_;
  size_t size_;
  size_t pos_;
};

/**
 * Serialize a dense tensor to buffer
 */
inline bool serialize_tensor(BufferWriter &writer, TensorDtype dtype,
                             std::span<const uint64_t> dims, const void *data) {
  TensorHeader header;
  header.dtype = dtype;
  header.ndim = static_cast<uint8_t>(dims.size());
  header.numel = 1;
  for (auto d : dims)
    header.numel *= d;

  if (!writer.write_value(header))
    return false;
  if (!writer.write_span(dims))
    return false;

  size_t data_size = header.numel * dtype_size(dtype);
  return writer.write(data, data_size);
}

/**
 * Serialize a sparse tensor (TopK compressed)
 */
inline bool serialize_sparse_tensor(BufferWriter &writer, TensorDtype dtype,
                                    std::span<const uint64_t> dims,
                                    std::span<const uint64_t> indices,
                                    const void *values) {
  SparseTensorHeader header;
  header.dtype = dtype;
  header.ndim = static_cast<uint8_t>(dims.size());
  header.original_numel = 1;
  for (auto d : dims)
    header.original_numel *= d;
  header.nnz = indices.size();

  if (!writer.write_value(header))
    return false;
  if (!writer.write_span(dims))
    return false;
  if (!writer.write_span(indices))
    return false;

  size_t values_size = header.nnz * dtype_size(dtype);
  return writer.write(values, values_size);
}

/**
 * Serialize multiple gradient tensors
 */
inline bool serialize_gradients(
    BufferWriter &writer, uint64_t step,
    const std::vector<
        std::tuple<TensorDtype, std::vector<uint64_t>, const void *>> &tensors,
    uint32_t compression_type = 0, float compression_ratio = 1.0f) {
  GradientHeader header;
  header.num_tensors = static_cast<uint32_t>(tensors.size());
  header.step = step;
  header.compression_type = compression_type;
  header.compression_ratio = compression_ratio;

  // Write header (will update total_size later)
  size_t header_pos = writer.position();
  if (!writer.write_value(header))
    return false;

  // Reserve space for offsets
  size_t offsets_size = tensors.size() * sizeof(uint64_t);
  uint8_t *offsets_ptr = writer.reserve(offsets_size);
  if (!offsets_ptr)
    return false;

  // Write tensors and record offsets
  std::vector<uint64_t> offsets;
  for (const auto &[dtype, dims, data] : tensors) {
    offsets.push_back(writer.position());
    if (!serialize_tensor(writer, dtype, dims, data))
      return false;
  }

  // Write offsets
  std::memcpy(offsets_ptr, offsets.data(), offsets_size);

  return true;
}

} // namespace gossip_rl::rpc
