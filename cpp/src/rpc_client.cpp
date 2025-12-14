#include "rpc/rpc_server.hpp"

#include <cstring>

namespace gossip_rl::rpc {

// RpcFuture implementation

RpcResponse RpcFuture::wait(std::chrono::milliseconds timeout) {
  std::unique_lock lock(state_->mutex);
  if (state_->completed) {
    return std::move(state_->response);
  }

  if (state_->cv.wait_for(lock, timeout,
                          [this] { return state_->completed; })) {
    return std::move(state_->response);
  }

  RpcResponse response;
  response.status = RpcStatus::DEADLINE_EXCEEDED;
  response.error_message = "Request timed out";
  return response;
}

bool RpcFuture::is_ready() const {
  std::lock_guard lock(state_->mutex);
  return state_->completed;
}

void RpcFuture::on_complete(std::function<void(RpcResponse)> callback) {
  std::lock_guard lock(state_->mutex);
  if (state_->completed) {
    callback(std::move(state_->response));
  } else {
    state_->callback = std::move(callback);
  }
}

void RpcFuture::cancel() {
  std::lock_guard lock(state_->mutex);
  state_->cancelled = true;
}

// RpcClient implementation

RpcClient::RpcClient(std::shared_ptr<EventLoop> event_loop,
                     std::shared_ptr<BufferPool> buffer_pool,
                     const RpcClientConfig &config)
    : event_loop_(std::move(event_loop)), buffer_pool_(std::move(buffer_pool)),
      config_(config) {

  transport_ = TransportFactory::create(config.transport, event_loop_,
                                        buffer_pool_, TransportConfig{});
}

RpcClient::~RpcClient() { disconnect(); }

bool RpcClient::connect(const Endpoint &endpoint) {
  connection_ = transport_->connect(endpoint);
  if (!connection_) {
    return false;
  }

  remote_endpoint_ = endpoint;
  connected_.store(true);

  transport_->set_message_callback(
      [this](Connection *conn, std::unique_ptr<Buffer> buffer) {
        on_message(conn, std::move(buffer));
      });

  transport_->set_disconnect_callback(
      [this](Connection *conn) { on_disconnect(conn); });

  spdlog::info("RPC client connected to {}:{}", endpoint.host, endpoint.port);
  return true;
}

void RpcClient::connect_async(const Endpoint &endpoint,
                              std::function<void(bool)> callback) {
  transport_->connect_async(
      endpoint, [this, endpoint, callback](Connection *conn, bool success) {
        if (success) {
          connection_ = transport_->get_connection(endpoint);
          remote_endpoint_ = endpoint;
          connected_.store(true);

          transport_->set_message_callback(
              [this](Connection *c, std::unique_ptr<Buffer> buffer) {
                on_message(c, std::move(buffer));
              });
        }
        if (callback)
          callback(success);
      });
}

bool RpcClient::is_connected() const {
  return connected_.load() && connection_ && connection_->is_healthy();
}

void RpcClient::disconnect() {
  if (connected_.exchange(false)) {
    if (connection_) {
      connection_->close();
      connection_.reset();
    }

    // Fail all pending requests
    std::lock_guard lock(pending_mutex_);
    for (auto &[id, future] : pending_requests_) {
      std::lock_guard flock(future.state_->mutex);
      future.state_->response.status = RpcStatus::UNAVAILABLE;
      future.state_->response.error_message = "Connection closed";
      future.state_->completed = true;
      future.state_->cv.notify_all();
      if (future.state_->callback) {
        future.state_->callback(std::move(future.state_->response));
      }
    }
    pending_requests_.clear();
  }
}

RpcResponse RpcClient::call(uint32_t method_id,
                            std::span<const uint8_t> payload,
                            std::chrono::milliseconds timeout) {
  if (timeout.count() == 0) {
    timeout = config_.default_timeout;
  }

  auto future = call_async(method_id, payload);
  return future.wait(timeout);
}

RpcFuture RpcClient::call_async(uint32_t method_id,
                                std::span<const uint8_t> payload) {
  if (!is_connected()) {
    RpcFuture future;
    std::lock_guard lock(future.state_->mutex);
    future.state_->response.status = RpcStatus::UNAVAILABLE;
    future.state_->response.error_message = "Not connected";
    future.state_->completed = true;
    return future;
  }

  auto buffer = buffer_pool_->acquire();
  if (!buffer) {
    RpcFuture future;
    std::lock_guard lock(future.state_->mutex);
    future.state_->response.status = RpcStatus::RESOURCE_EXHAUSTED;
    future.state_->response.error_message = "Buffer pool exhausted";
    future.state_->completed = true;
    return future;
  }

  uint64_t request_id = next_request_id_.fetch_add(1);

  // Write header
  RpcHeader header;
  header.request_id = request_id;
  header.method_id = method_id;
  header.payload_size = static_cast<uint32_t>(payload.size());
  header.flags = 0;

  buffer->write(&header, sizeof(header));
  buffer->write(payload.data(), payload.size());

  // Create future and register
  RpcFuture future;
  {
    std::lock_guard lock(pending_mutex_);
    pending_requests_[request_id] = future;
  }

  // Send request
  connection_->send(std::move(buffer), [this, request_id](bool success) {
    if (!success) {
      std::lock_guard lock(pending_mutex_);
      auto it = pending_requests_.find(request_id);
      if (it != pending_requests_.end()) {
        std::lock_guard flock(it->second.state_->mutex);
        it->second.state_->response.status = RpcStatus::INTERNAL;
        it->second.state_->response.error_message = "Send failed";
        it->second.state_->completed = true;
        it->second.state_->cv.notify_all();
        pending_requests_.erase(it);
      }
    }
  });

  return future;
}

bool RpcClient::call_oneway(uint32_t method_id,
                            std::span<const uint8_t> payload) {
  if (!is_connected()) {
    return false;
  }

  auto buffer = buffer_pool_->acquire();
  if (!buffer) {
    return false;
  }

  RpcHeader header;
  header.request_id = 0; // No response expected
  header.method_id = method_id;
  header.payload_size = static_cast<uint32_t>(payload.size());
  header.flags = RpcFlags::ONE_WAY;

  buffer->write(&header, sizeof(header));
  buffer->write(payload.data(), payload.size());

  return connection_->send(std::move(buffer));
}

void RpcClient::on_message(Connection *conn, std::unique_ptr<Buffer> buffer) {
  (void)conn;

  if (buffer->size() < RpcHeader::SIZE) {
    spdlog::warn("RPC response too small");
    return;
  }

  RpcHeader header;
  std::memcpy(&header, buffer->data(), sizeof(header));

  // Find pending request
  RpcFuture future;
  {
    std::lock_guard lock(pending_mutex_);
    auto it = pending_requests_.find(header.request_id);
    if (it == pending_requests_.end()) {
      spdlog::warn("Unknown request ID: {}", header.request_id);
      return;
    }
    future = std::move(it->second);
    pending_requests_.erase(it);
  }

  // Parse response
  RpcResponse response;
  response.status = static_cast<RpcStatus>(header.method_id);

  // Read error message
  size_t offset = RpcHeader::SIZE;
  if (buffer->size() > offset + sizeof(uint32_t)) {
    uint32_t error_len;
    std::memcpy(&error_len, buffer->data() + offset, sizeof(error_len));
    offset += sizeof(uint32_t);

    if (error_len > 0 && buffer->size() >= offset + error_len) {
      response.error_message.assign(
          reinterpret_cast<const char *>(buffer->data() + offset), error_len);
      offset += error_len;
    }
  }

  // Payload
  if (buffer->size() > offset) {
    auto payload_buf = buffer_pool_->acquire();
    if (payload_buf) {
      payload_buf->write(buffer->data() + offset, buffer->size() - offset);
      response.payload = std::move(payload_buf);
    }
  }

  // Complete the future
  {
    std::lock_guard lock(future.state_->mutex);
    future.state_->response = std::move(response);
    future.state_->completed = true;
    future.state_->cv.notify_all();

    if (future.state_->callback) {
      future.state_->callback(std::move(future.state_->response));
    }
  }
}

void RpcClient::on_disconnect(Connection *conn) {
  (void)conn;
  connected_.store(false);
  spdlog::info("RPC client disconnected from {}:{}", remote_endpoint_.host,
               remote_endpoint_.port);
}

// RpcConnectionPool implementation

RpcConnectionPool::RpcConnectionPool(std::shared_ptr<EventLoop> event_loop,
                                     std::shared_ptr<BufferPool> buffer_pool,
                                     const Config &config)
    : event_loop_(std::move(event_loop)), buffer_pool_(std::move(buffer_pool)),
      config_(config) {

  if (config_.enable_health_check) {
    health_check_thread_ = std::thread([this] { health_check_loop(); });
  }
}

RpcConnectionPool::RpcConnectionPool(std::shared_ptr<EventLoop> event_loop,
                                     std::shared_ptr<BufferPool> buffer_pool)
    : RpcConnectionPool(std::move(event_loop), std::move(buffer_pool),
                        Config{}) {}

RpcConnectionPool::~RpcConnectionPool() {
  running_.store(false);
  if (health_check_thread_.joinable()) {
    health_check_thread_.join();
  }
  close_all();
}

std::shared_ptr<RpcClient>
RpcConnectionPool::get_client(const Endpoint &endpoint) {
  std::string key = endpoint.to_string();

  std::lock_guard lock(pools_mutex_);
  auto &pool = pools_[key];

  // Find an available client
  for (auto &client : pool) {
    if (client->is_connected()) {
      return client;
    }
  }

  // Create new client if under limit
  if (pool.size() < config_.max_connections_per_endpoint) {
    auto client = std::make_shared<RpcClient>(event_loop_, buffer_pool_);
    if (client->connect(endpoint)) {
      pool.push_back(client);
      return client;
    }
  }

  return nullptr;
}

void RpcConnectionPool::release(std::shared_ptr<RpcClient> client) {
  // Client stays in pool, no action needed
  (void)client;
}

void RpcConnectionPool::close_endpoint(const Endpoint &endpoint) {
  std::string key = endpoint.to_string();

  std::lock_guard lock(pools_mutex_);
  auto it = pools_.find(key);
  if (it != pools_.end()) {
    for (auto &client : it->second) {
      client->disconnect();
    }
    pools_.erase(it);
  }
}

void RpcConnectionPool::close_all() {
  std::lock_guard lock(pools_mutex_);
  for (auto &[key, pool] : pools_) {
    for (auto &client : pool) {
      client->disconnect();
    }
  }
  pools_.clear();
}

void RpcConnectionPool::health_check_loop() {
  while (running_.load()) {
    std::this_thread::sleep_for(config_.health_check_interval);

    std::lock_guard lock(pools_mutex_);
    for (auto &[key, pool] : pools_) {
      pool.erase(std::remove_if(pool.begin(), pool.end(),
                                [](const std::shared_ptr<RpcClient> &client) {
                                  return !client->is_connected();
                                }),
                 pool.end());
    }
  }
}

} // namespace gossip_rl::rpc
