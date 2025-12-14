#include "rpc/transport.hpp"

#include <thread>

namespace gossip_rl::rpc {

/**
 * Shared Memory Connection implementation
 */
class SharedMemoryTransport::ShmConnection : public Connection {
public:
  ShmConnection(SharedMemoryTransport *transport,
                std::unique_ptr<SharedMemoryRegion> send_region,
                std::unique_ptr<SharedMemoryRegion> recv_region,
                const Endpoint &local, const Endpoint &remote, uint64_t id)
      : transport_(transport), send_region_(std::move(send_region)),
        recv_region_(std::move(recv_region)), local_endpoint_(local),
        remote_endpoint_(remote), id_(id), state_(ConnectionState::CONNECTED) {

    stats_.connected_at = std::chrono::steady_clock::now();
    stats_.last_activity = stats_.connected_at;

    // Initialize ring buffers (header + data area)
    constexpr size_t RING_CAPACITY = 1024;
    constexpr size_t ENTRY_SIZE = 65536; // 64KB per message

    if (send_region_) {
      send_ring_ = std::make_unique<SharedRingBuffer<RingEntry>>(
          send_region_.get(), RING_CAPACITY);
    }
    if (recv_region_) {
      recv_ring_ = std::make_unique<SharedRingBuffer<RingEntry>>(
          recv_region_.get(), RING_CAPACITY);
    }
  }

  bool send(std::unique_ptr<Buffer> buffer) override {
    return send(std::move(buffer), nullptr);
  }

  bool send(std::unique_ptr<Buffer> buffer,
            std::function<void(bool)> callback) override {
    if (!send_ring_ || state_ != ConnectionState::CONNECTED) {
      if (callback)
        callback(false);
      return false;
    }

    RingEntry entry;
    entry.size = buffer->size();
    std::memcpy(entry.data, buffer->data(),
                std::min(buffer->size(), sizeof(entry.data)));

    if (send_ring_->try_push(entry)) {
      stats_.bytes_sent.fetch_add(entry.size, std::memory_order_relaxed);
      stats_.messages_sent.fetch_add(1, std::memory_order_relaxed);
      stats_.last_activity = std::chrono::steady_clock::now();

      transport_->buffer_pool_->release(std::move(buffer));
      if (callback)
        callback(true);
      return true;
    }

    if (callback)
      callback(false);
    return false;
  }

  void close() override {
    ConnectionState expected = ConnectionState::CONNECTED;
    if (state_.compare_exchange_strong(expected, ConnectionState::CLOSED)) {
      send_region_.reset();
      recv_region_.reset();
    }
  }

  [[nodiscard]] ConnectionState state() const override {
    return state_.load(std::memory_order_acquire);
  }

  [[nodiscard]] const Endpoint &remote_endpoint() const override {
    return remote_endpoint_;
  }

  [[nodiscard]] const Endpoint &local_endpoint() const override {
    return local_endpoint_;
  }

  [[nodiscard]] TransportType transport_type() const override {
    return TransportType::SHARED_MEMORY;
  }

  [[nodiscard]] const ConnectionStats &stats() const override { return stats_; }

  [[nodiscard]] bool is_healthy() const override {
    return state_.load() == ConnectionState::CONNECTED;
  }

  [[nodiscard]] uint64_t id() const override { return id_; }

  void set_user_data(void *data) override { user_data_ = data; }

  [[nodiscard]] void *user_data() const override { return user_data_; }

  // Poll for incoming messages
  bool poll_recv(std::unique_ptr<Buffer> &buffer) {
    if (!recv_ring_)
      return false;

    RingEntry entry;
    if (recv_ring_->try_pop(entry)) {
      buffer = transport_->buffer_pool_->acquire();
      if (!buffer)
        return false;

      buffer->write(entry.data, entry.size);
      stats_.bytes_received.fetch_add(entry.size, std::memory_order_relaxed);
      stats_.messages_received.fetch_add(1, std::memory_order_relaxed);
      stats_.last_activity = std::chrono::steady_clock::now();
      return true;
    }
    return false;
  }

private:
  struct RingEntry {
    uint32_t size;
    uint8_t data[65532]; // 64KB - 4 bytes for size
  };

  SharedMemoryTransport *transport_;
  std::unique_ptr<SharedMemoryRegion> send_region_;
  std::unique_ptr<SharedMemoryRegion> recv_region_;
  std::unique_ptr<SharedRingBuffer<RingEntry>> send_ring_;
  std::unique_ptr<SharedRingBuffer<RingEntry>> recv_ring_;
  Endpoint local_endpoint_;
  Endpoint remote_endpoint_;
  uint64_t id_;
  std::atomic<ConnectionState> state_;
  ConnectionStats stats_;
  void *user_data_{nullptr};
};

// SharedMemoryTransport implementation

SharedMemoryTransport::SharedMemoryTransport(
    std::shared_ptr<BufferPool> buffer_pool, const std::string &name_prefix)
    : buffer_pool_(std::move(buffer_pool)), name_prefix_(name_prefix) {}

SharedMemoryTransport::~SharedMemoryTransport() { stop(); }

bool SharedMemoryTransport::listen(const Endpoint &endpoint) {
  // For shared memory, "listening" means creating a well-known region
  // that clients can connect to
  running_.store(true);
  spdlog::info("Shared memory transport listening for endpoint {}",
               endpoint.to_string());
  return true;
}

std::shared_ptr<Connection>
SharedMemoryTransport::connect(const Endpoint &endpoint) {
  // Create paired shared memory regions for bidirectional communication
  std::string base_name =
      name_prefix_ + "_" + endpoint.host + "_" + std::to_string(endpoint.port);

  constexpr size_t REGION_SIZE = 128 * 1024 * 1024; // 128MB

  // Create send region (client -> server)
  auto send_region = SharedMemoryRegion::create(
      base_name + "_c2s_" + std::to_string(next_connection_id_.load()),
      REGION_SIZE);
  if (!send_region) {
    return nullptr;
  }

  // Create recv region (server -> client)
  auto recv_region = SharedMemoryRegion::create(
      base_name + "_s2c_" + std::to_string(next_connection_id_.load()),
      REGION_SIZE);
  if (!recv_region) {
    return nullptr;
  }

  uint64_t conn_id = next_connection_id_.fetch_add(1);
  Endpoint local{"localhost", 0};

  auto conn = std::make_shared<ShmConnection>(this, std::move(send_region),
                                              std::move(recv_region), local,
                                              endpoint, conn_id);

  {
    std::lock_guard lock(connections_mutex_);
    connections_[conn_id] = conn;
  }

  spdlog::info("Created shared memory connection to {}", endpoint.to_string());
  return conn;
}

void SharedMemoryTransport::connect_async(const Endpoint &endpoint,
                                          ConnectCallback callback) {
  // Shared memory operations are fast enough to do synchronously
  auto conn = connect(endpoint);
  if (callback) {
    callback(conn.get(), conn != nullptr);
  }
}

void SharedMemoryTransport::stop() {
  running_.store(false);

  std::lock_guard lock(connections_mutex_);
  for (auto &[id, conn] : connections_) {
    conn->close();
  }
  connections_.clear();
}

void SharedMemoryTransport::set_message_callback(MessageCallback callback) {
  message_callback_ = std::move(callback);
}

void SharedMemoryTransport::set_disconnect_callback(
    DisconnectCallback callback) {
  disconnect_callback_ = std::move(callback);
}

void SharedMemoryTransport::set_error_callback(ErrorCallback callback) {
  error_callback_ = std::move(callback);
}

std::vector<std::shared_ptr<Connection>>
SharedMemoryTransport::connections() const {
  std::lock_guard lock(connections_mutex_);
  std::vector<std::shared_ptr<Connection>> result;
  result.reserve(connections_.size());
  for (const auto &[id, conn] : connections_) {
    result.push_back(conn);
  }
  return result;
}

std::shared_ptr<Connection>
SharedMemoryTransport::get_connection(const Endpoint &endpoint) const {
  std::lock_guard lock(connections_mutex_);
  for (const auto &[id, conn] : connections_) {
    if (conn->remote_endpoint() == endpoint) {
      return conn;
    }
  }
  return nullptr;
}

void SharedMemoryTransport::poll() {
  if (!running_.load())
    return;

  std::lock_guard lock(connections_mutex_);
  for (auto &[id, conn] : connections_) {
    std::unique_ptr<Buffer> buffer;
    while (conn->poll_recv(buffer)) {
      if (message_callback_) {
        message_callback_(conn.get(), std::move(buffer));
      }
    }
  }
}

} // namespace gossip_rl::rpc
