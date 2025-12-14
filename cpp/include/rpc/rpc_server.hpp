#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rpc/buffer_pool.hpp"
#include "rpc/event_loop.hpp"
#include "rpc/transport.hpp"

namespace gossip_rl::rpc {

// Forward declarations
class RpcServer;
class RpcClient;
class RpcChannel;

/**
 * RPC message header - fixed size for zero-copy parsing
 */
struct alignas(8) RpcHeader {
  uint32_t magic = 0x47525043; // "GRPC"
  uint32_t version = 1;
  uint64_t request_id;
  uint32_t method_id;
  uint32_t payload_size;
  uint32_t flags;
  uint32_t reserved;

  static constexpr size_t SIZE = 32;
};

static_assert(sizeof(RpcHeader) == RpcHeader::SIZE, "RpcHeader size mismatch");

/**
 * RPC message flags
 */
namespace RpcFlags {
constexpr uint32_t NONE = 0;
constexpr uint32_t COMPRESSED = 1 << 0;
constexpr uint32_t ENCRYPTED = 1 << 1;
constexpr uint32_t ONE_WAY = 1 << 2; // No response expected
constexpr uint32_t PRIORITY = 1 << 3;
constexpr uint32_t STREAMING = 1 << 4;
} // namespace RpcFlags

/**
 * RPC status codes
 */
enum class RpcStatus : uint8_t {
  OK = 0,
  CANCELLED = 1,
  INVALID_ARGUMENT = 2,
  DEADLINE_EXCEEDED = 3,
  NOT_FOUND = 4,
  ALREADY_EXISTS = 5,
  PERMISSION_DENIED = 6,
  RESOURCE_EXHAUSTED = 7,
  INTERNAL = 8,
  UNAVAILABLE = 9,
  UNKNOWN = 255
};

/**
 * RPC request context
 */
struct RpcContext {
  uint64_t request_id{0};
  uint32_t method_id{0};
  std::chrono::steady_clock::time_point deadline;
  std::shared_ptr<Connection> connection;
  std::unordered_map<std::string, std::string> metadata;
  bool cancelled{false};

  [[nodiscard]] bool is_expired() const {
    return std::chrono::steady_clock::now() > deadline;
  }
};

/**
 * RPC response
 */
struct RpcResponse {
  RpcStatus status{RpcStatus::OK};
  std::string error_message;
  std::unique_ptr<Buffer> payload;

  [[nodiscard]] bool ok() const { return status == RpcStatus::OK; }
};

/**
 * Method handler type
 */
using RpcHandler =
    std::function<RpcResponse(const RpcContext &, std::span<const uint8_t>)>;

/**
 * Async method handler type (for streaming and long-running operations)
 */
using AsyncRpcHandler =
    std::function<void(const RpcContext &, std::span<const uint8_t>,
                       std::function<void(RpcResponse)>)>;

/**
 * RPC method descriptor
 */
struct RpcMethod {
  uint32_t id;
  std::string name;
  RpcHandler handler;
  AsyncRpcHandler async_handler;
  bool is_async{false};
  std::chrono::milliseconds default_timeout{5000};
};

/**
 * RPC service descriptor
 */
struct RpcService {
  std::string name;
  std::unordered_map<uint32_t, RpcMethod> methods;

  void register_method(
      uint32_t id, const std::string &name, RpcHandler handler,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
    methods[id] =
        RpcMethod{id, name, std::move(handler), nullptr, false, timeout};
  }

  void register_async_method(
      uint32_t id, const std::string &name, AsyncRpcHandler handler,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
    methods[id] =
        RpcMethod{id, name, nullptr, std::move(handler), true, timeout};
  }
};

/**
 * RPC Server configuration
 */
struct RpcServerConfig {
  Endpoint listen_endpoint;
  TransportType transport = TransportType::AUTO;
  size_t max_concurrent_requests = 10000;
  std::chrono::milliseconds default_timeout{5000};
  size_t worker_threads = 0; // 0 = use event loop thread
  bool enable_compression = false;
};

/**
 * RPC Server statistics
 */
struct RpcServerStats {
  std::atomic<uint64_t> total_requests{0};
  std::atomic<uint64_t> successful_requests{0};
  std::atomic<uint64_t> failed_requests{0};
  std::atomic<uint64_t> active_requests{0};
  std::atomic<uint64_t> total_bytes_received{0};
  std::atomic<uint64_t> total_bytes_sent{0};
  std::atomic<uint64_t> avg_latency_us{0};

  void reset() {
    total_requests = 0;
    successful_requests = 0;
    failed_requests = 0;
    active_requests = 0;
    total_bytes_received = 0;
    total_bytes_sent = 0;
    avg_latency_us = 0;
  }
};

/**
 * High-performance RPC Server
 */
class RpcServer {
public:
  RpcServer(std::shared_ptr<EventLoop> event_loop,
            std::shared_ptr<BufferPool> buffer_pool,
            const RpcServerConfig &config);
  ~RpcServer();

  // Non-copyable
  RpcServer(const RpcServer &) = delete;
  RpcServer &operator=(const RpcServer &) = delete;

  /**
   * Register a service
   */
  void register_service(std::shared_ptr<RpcService> service);

  /**
   * Start the server
   */
  bool start();

  /**
   * Stop the server
   */
  void stop();

  /**
   * Check if server is running
   */
  [[nodiscard]] bool is_running() const {
    return running_.load(std::memory_order_acquire);
  }

  /**
   * Get server statistics
   */
  [[nodiscard]] const RpcServerStats &stats() const { return stats_; }

  /**
   * Get listen endpoint
   */
  [[nodiscard]] const Endpoint &endpoint() const {
    return config_.listen_endpoint;
  }

private:
  void on_message(Connection *conn, std::unique_ptr<Buffer> buffer);
  void on_disconnect(Connection *conn);
  void process_request(const RpcContext &ctx, std::span<const uint8_t> payload);
  void send_response(Connection *conn, uint64_t request_id,
                     const RpcResponse &response);

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;
  std::unique_ptr<Transport> transport_;
  RpcServerConfig config_;
  RpcServerStats stats_;

  std::unordered_map<std::string, std::shared_ptr<RpcService>> services_;
  std::unordered_map<uint32_t, RpcMethod *> method_lookup_;
  std::mutex services_mutex_;

  std::atomic<bool> running_{false};
};

/**
 * RPC Client configuration
 */
struct RpcClientConfig {
  TransportType transport = TransportType::AUTO;
  std::chrono::milliseconds connect_timeout{5000};
  std::chrono::milliseconds default_timeout{5000};
  size_t max_pending_requests = 1000;
  bool enable_compression = false;
  size_t retry_count = 3;
  std::chrono::milliseconds retry_delay{100};
};

/**
 * RPC call future for async operations
 */
class RpcFuture {
public:
  RpcFuture() = default;

  /**
   * Wait for the response with timeout
   */
  [[nodiscard]] RpcResponse
  wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{5000});

  /**
   * Check if result is ready
   */
  [[nodiscard]] bool is_ready() const;

  /**
   * Set completion callback
   */
  void on_complete(std::function<void(RpcResponse)> callback);

  /**
   * Cancel the request
   */
  void cancel();

private:
  friend class RpcClient;
  friend class RpcTest;

  struct State {
    std::mutex mutex;
    std::condition_variable cv;
    RpcResponse response;
    std::function<void(RpcResponse)> callback;
    bool completed{false};
    bool cancelled{false};
  };

  std::shared_ptr<State> state_ = std::make_shared<State>();
};

/**
 * RPC Client for connecting to servers
 */
class RpcClient {
public:
  RpcClient(std::shared_ptr<EventLoop> event_loop,
            std::shared_ptr<BufferPool> buffer_pool,
            const RpcClientConfig &config = RpcClientConfig{});
  ~RpcClient();

  // Non-copyable
  RpcClient(const RpcClient &) = delete;
  RpcClient &operator=(const RpcClient &) = delete;

  /**
   * Connect to a server
   */
  bool connect(const Endpoint &endpoint);

  /**
   * Connect asynchronously
   */
  void connect_async(const Endpoint &endpoint,
                     std::function<void(bool)> callback);

  /**
   * Check if connected
   */
  [[nodiscard]] bool is_connected() const;

  /**
   * Disconnect from server
   */
  void disconnect();

  /**
   * Call an RPC method synchronously
   */
  RpcResponse
  call(uint32_t method_id, std::span<const uint8_t> payload,
       std::chrono::milliseconds timeout = std::chrono::milliseconds{0});

  /**
   * Call an RPC method asynchronously
   */
  RpcFuture call_async(uint32_t method_id, std::span<const uint8_t> payload);

  /**
   * One-way call (no response expected)
   */
  bool call_oneway(uint32_t method_id, std::span<const uint8_t> payload);

  /**
   * Get the remote endpoint
   */
  [[nodiscard]] const Endpoint &remote_endpoint() const {
    return remote_endpoint_;
  }

private:
  void on_message(Connection *conn, std::unique_ptr<Buffer> buffer);
  void on_disconnect(Connection *conn);

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;
  std::unique_ptr<Transport> transport_;
  std::shared_ptr<Connection> connection_;
  RpcClientConfig config_;
  Endpoint remote_endpoint_;

  std::unordered_map<uint64_t, RpcFuture> pending_requests_;
  std::mutex pending_mutex_;

  std::atomic<uint64_t> next_request_id_{1};
  std::atomic<bool> connected_{false};
};

/**
 * Connection pool for managing multiple client connections
 */
class RpcConnectionPool {
public:
  struct Config {
    size_t max_connections_per_endpoint = 4;
    std::chrono::milliseconds idle_timeout{60000};
    bool enable_health_check = true;
    std::chrono::milliseconds health_check_interval{10000};
  };

  RpcConnectionPool(std::shared_ptr<EventLoop> event_loop,
                    std::shared_ptr<BufferPool> buffer_pool,
                    const Config &config);
  RpcConnectionPool(std::shared_ptr<EventLoop> event_loop,
                    std::shared_ptr<BufferPool> buffer_pool);
  ~RpcConnectionPool();

  /**
   * Get or create a client for the given endpoint
   */
  std::shared_ptr<RpcClient> get_client(const Endpoint &endpoint);

  /**
   * Release a client back to the pool
   */
  void release(std::shared_ptr<RpcClient> client);

  /**
   * Close all connections to an endpoint
   */
  void close_endpoint(const Endpoint &endpoint);

  /**
   * Close all connections
   */
  void close_all();

private:
  void health_check_loop();

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;
  Config config_;

  std::unordered_map<std::string, std::vector<std::shared_ptr<RpcClient>>>
      pools_;
  std::mutex pools_mutex_;

  std::atomic<bool> running_{true};
  std::thread health_check_thread_;
};

} // namespace gossip_rl::rpc
