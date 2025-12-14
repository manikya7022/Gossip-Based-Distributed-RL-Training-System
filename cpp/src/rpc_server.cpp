#include "rpc/rpc_server.hpp"

#include <cstring>

namespace gossip_rl::rpc {

RpcServer::RpcServer(std::shared_ptr<EventLoop> event_loop,
                     std::shared_ptr<BufferPool> buffer_pool,
                     const RpcServerConfig &config)
    : event_loop_(std::move(event_loop)), buffer_pool_(std::move(buffer_pool)),
      config_(config) {

  transport_ = TransportFactory::create(
      config.transport, event_loop_, buffer_pool_,
      TransportConfig{.max_connections = config.max_concurrent_requests,
                      .connect_timeout = config.default_timeout});
}

RpcServer::~RpcServer() { stop(); }

void RpcServer::register_service(std::shared_ptr<RpcService> service) {
  std::lock_guard lock(services_mutex_);
  services_[service->name] = service;

  // Build method lookup table
  for (auto &[id, method] : service->methods) {
    method_lookup_[id] = &method;
  }

  spdlog::info("Registered RPC service '{}' with {} methods", service->name,
               service->methods.size());
}

bool RpcServer::start() {
  if (running_.load()) {
    return false;
  }

  // Set up callbacks
  transport_->set_message_callback(
      [this](Connection *conn, std::unique_ptr<Buffer> buffer) {
        on_message(conn, std::move(buffer));
      });

  transport_->set_disconnect_callback(
      [this](Connection *conn) { on_disconnect(conn); });

  transport_->set_error_callback(
      [this](Connection *conn, const std::string &error) {
        spdlog::error("RPC connection error: {}", error);
        (void)conn;
      });

  if (!transport_->listen(config_.listen_endpoint)) {
    return false;
  }

  running_.store(true);
  spdlog::info("RPC server started on {}:{}", config_.listen_endpoint.host,
               config_.listen_endpoint.port);
  return true;
}

void RpcServer::stop() {
  if (running_.exchange(false)) {
    transport_->stop();
    spdlog::info("RPC server stopped");
  }
}

void RpcServer::on_message(Connection *conn, std::unique_ptr<Buffer> buffer) {
  stats_.total_requests.fetch_add(1, std::memory_order_relaxed);
  stats_.active_requests.fetch_add(1, std::memory_order_relaxed);
  stats_.total_bytes_received.fetch_add(buffer->size(),
                                        std::memory_order_relaxed);

  auto start_time = std::chrono::steady_clock::now();

  // Parse header
  if (buffer->size() < RpcHeader::SIZE) {
    spdlog::warn("RPC message too small: {} bytes", buffer->size());
    stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
    stats_.active_requests.fetch_sub(1, std::memory_order_relaxed);
    return;
  }

  RpcHeader header;
  std::memcpy(&header, buffer->data(), sizeof(header));

  if (header.magic != 0x47525043) {
    spdlog::warn("Invalid RPC magic: {:#x}", header.magic);
    stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
    stats_.active_requests.fetch_sub(1, std::memory_order_relaxed);
    return;
  }

  // Create context
  RpcContext ctx;
  ctx.request_id = header.request_id;
  ctx.method_id = header.method_id;
  ctx.deadline = std::chrono::steady_clock::now() + config_.default_timeout;
  ctx.connection = transport_->get_connection(conn->remote_endpoint());

  // Get payload
  std::span<const uint8_t> payload(buffer->data() + RpcHeader::SIZE,
                                   buffer->size() - RpcHeader::SIZE);

  // Find and execute method
  RpcMethod *method = nullptr;
  {
    std::lock_guard lock(services_mutex_);
    auto it = method_lookup_.find(header.method_id);
    if (it != method_lookup_.end()) {
      method = it->second;
    }
  }

  if (!method) {
    RpcResponse response;
    response.status = RpcStatus::NOT_FOUND;
    response.error_message =
        "Method not found: " + std::to_string(header.method_id);
    send_response(conn, header.request_id, response);
    stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
    stats_.active_requests.fetch_sub(1, std::memory_order_relaxed);
    return;
  }

  // Execute handler
  RpcResponse response;
  if (method->is_async && method->async_handler) {
    method->async_handler(
        ctx, payload,
        [this, conn, request_id = header.request_id,
         start_time](RpcResponse resp) {
          send_response(conn, request_id, resp);

          auto duration = std::chrono::steady_clock::now() - start_time;
          auto us =
              std::chrono::duration_cast<std::chrono::microseconds>(duration)
                  .count();
          stats_.avg_latency_us.store(us, std::memory_order_relaxed);

          if (resp.ok()) {
            stats_.successful_requests.fetch_add(1, std::memory_order_relaxed);
          } else {
            stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
          }
          stats_.active_requests.fetch_sub(1, std::memory_order_relaxed);
        });
  } else if (method->handler) {
    response = method->handler(ctx, payload);
    send_response(conn, header.request_id, response);

    auto duration = std::chrono::steady_clock::now() - start_time;
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    stats_.avg_latency_us.store(us, std::memory_order_relaxed);

    if (response.ok()) {
      stats_.successful_requests.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
    }
    stats_.active_requests.fetch_sub(1, std::memory_order_relaxed);
  }
}

void RpcServer::on_disconnect(Connection *conn) {
  spdlog::debug("Client disconnected: {}", conn->remote_endpoint().to_string());
}

void RpcServer::send_response(Connection *conn, uint64_t request_id,
                              const RpcResponse &response) {
  auto buffer = buffer_pool_->acquire();
  if (!buffer) {
    spdlog::error("Failed to acquire buffer for response");
    return;
  }

  // Write header
  RpcHeader header;
  header.request_id = request_id;
  header.method_id = static_cast<uint32_t>(response.status);
  header.payload_size = response.payload ? response.payload->size() : 0;
  header.flags = 0;

  buffer->write(&header, sizeof(header));

  // Write error message if present
  if (!response.error_message.empty()) {
    uint32_t len = static_cast<uint32_t>(response.error_message.size());
    buffer->write(&len, sizeof(len));
    buffer->write(response.error_message.data(), response.error_message.size());
  } else {
    uint32_t len = 0;
    buffer->write(&len, sizeof(len));
  }

  // Write payload
  if (response.payload) {
    buffer->write(response.payload->data(), response.payload->size());
  }

  stats_.total_bytes_sent.fetch_add(buffer->size(), std::memory_order_relaxed);

  conn->send(std::move(buffer));
}

} // namespace gossip_rl::rpc
