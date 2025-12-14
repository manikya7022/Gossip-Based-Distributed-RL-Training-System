#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rpc/buffer_pool.hpp"
#include "rpc/event_loop.hpp"

namespace gossip_rl::rpc {

// Forward declarations
class Transport;
class Connection;
class TcpTransport;
class SharedMemoryTransport;

/**
 * Transport type enumeration
 */
enum class TransportType : uint8_t {
  TCP,
  SHARED_MEMORY,
  AUTO // Automatically select based on peer location
};

/**
 * Connection state
 */
enum class ConnectionState : uint8_t {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  CLOSING,
  CLOSED,
  ERROR
};

/**
 * Peer endpoint descriptor
 */
struct Endpoint {
  std::string host;
  uint16_t port{0};
  TransportType preferred_transport{TransportType::AUTO};

  [[nodiscard]] std::string to_string() const {
    return host + ":" + std::to_string(port);
  }

  bool operator==(const Endpoint &other) const {
    return host == other.host && port == other.port;
  }
};

// Hash for Endpoint
struct EndpointHash {
  size_t operator()(const Endpoint &ep) const {
    return std::hash<std::string>{}(ep.to_string());
  }
};

/**
 * Connection statistics
 */
struct ConnectionStats {
  std::atomic<uint64_t> bytes_sent{0};
  std::atomic<uint64_t> bytes_received{0};
  std::atomic<uint64_t> messages_sent{0};
  std::atomic<uint64_t> messages_received{0};
  std::atomic<uint64_t> errors{0};
  std::chrono::steady_clock::time_point connected_at;
  std::chrono::steady_clock::time_point last_activity;
};

/**
 * Transport callback types
 */
using MessageCallback =
    std::function<void(Connection *, std::unique_ptr<Buffer>)>;
using ConnectCallback = std::function<void(Connection *, bool success)>;
using DisconnectCallback = std::function<void(Connection *)>;
using ErrorCallback =
    std::function<void(Connection *, const std::string &error)>;

/**
 * Abstract connection interface
 */
class Connection {
public:
  virtual ~Connection() = default;

  /**
   * Send data asynchronously
   * @return true if send was queued successfully
   */
  virtual bool send(std::unique_ptr<Buffer> buffer) = 0;

  /**
   * Send data with callback
   */
  virtual bool send(std::unique_ptr<Buffer> buffer,
                    std::function<void(bool success)> callback) = 0;

  /**
   * Close the connection
   */
  virtual void close() = 0;

  /**
   * Get connection state
   */
  [[nodiscard]] virtual ConnectionState state() const = 0;

  /**
   * Get remote endpoint
   */
  [[nodiscard]] virtual const Endpoint &remote_endpoint() const = 0;

  /**
   * Get local endpoint
   */
  [[nodiscard]] virtual const Endpoint &local_endpoint() const = 0;

  /**
   * Get transport type
   */
  [[nodiscard]] virtual TransportType transport_type() const = 0;

  /**
   * Get connection statistics
   */
  [[nodiscard]] virtual const ConnectionStats &stats() const = 0;

  /**
   * Check if connection is healthy
   */
  [[nodiscard]] virtual bool is_healthy() const = 0;

  /**
   * Get connection ID
   */
  [[nodiscard]] virtual uint64_t id() const = 0;

  /**
   * Set user data
   */
  virtual void set_user_data(void *data) = 0;
  [[nodiscard]] virtual void *user_data() const = 0;
};

/**
 * Transport configuration
 */
struct TransportConfig {
  size_t max_connections = 1000;
  size_t recv_buffer_size = 64 * 1024;
  size_t send_buffer_size = 64 * 1024;
  std::chrono::milliseconds connect_timeout{5000};
  std::chrono::milliseconds idle_timeout{30000};
  bool tcp_nodelay = true;
  bool tcp_keepalive = true;
  int tcp_keepalive_idle = 60;
  int tcp_keepalive_interval = 10;
  int tcp_keepalive_count = 5;
  bool reuse_addr = true;
  bool reuse_port = true;
};

/**
 * Abstract transport interface
 */
class Transport {
public:
  virtual ~Transport() = default;

  /**
   * Start listening on the specified endpoint
   */
  virtual bool listen(const Endpoint &endpoint) = 0;

  /**
   * Connect to a remote endpoint
   */
  virtual std::shared_ptr<Connection> connect(const Endpoint &endpoint) = 0;

  /**
   * Connect with callback
   */
  virtual void connect_async(const Endpoint &endpoint,
                             ConnectCallback callback) = 0;

  /**
   * Stop the transport
   */
  virtual void stop() = 0;

  /**
   * Get transport type
   */
  [[nodiscard]] virtual TransportType type() const = 0;

  /**
   * Set message callback
   */
  virtual void set_message_callback(MessageCallback callback) = 0;

  /**
   * Set disconnect callback
   */
  virtual void set_disconnect_callback(DisconnectCallback callback) = 0;

  /**
   * Set error callback
   */
  virtual void set_error_callback(ErrorCallback callback) = 0;

  /**
   * Get all active connections
   */
  [[nodiscard]] virtual std::vector<std::shared_ptr<Connection>>
  connections() const = 0;

  /**
   * Get connection by endpoint
   */
  [[nodiscard]] virtual std::shared_ptr<Connection>
  get_connection(const Endpoint &endpoint) const = 0;
};

/**
 * TCP Transport implementation
 */
class TcpTransport : public Transport {
public:
  TcpTransport(std::shared_ptr<EventLoop> event_loop,
               std::shared_ptr<BufferPool> buffer_pool,
               const TransportConfig &config = TransportConfig{});
  ~TcpTransport() override;

  bool listen(const Endpoint &endpoint) override;
  std::shared_ptr<Connection> connect(const Endpoint &endpoint) override;
  void connect_async(const Endpoint &endpoint,
                     ConnectCallback callback) override;
  void stop() override;

  [[nodiscard]] TransportType type() const override {
    return TransportType::TCP;
  }

  void set_message_callback(MessageCallback callback) override;
  void set_disconnect_callback(DisconnectCallback callback) override;
  void set_error_callback(ErrorCallback callback) override;

  [[nodiscard]] std::vector<std::shared_ptr<Connection>>
  connections() const override;
  [[nodiscard]] std::shared_ptr<Connection>
  get_connection(const Endpoint &endpoint) const override;

private:
  class TcpConnection;

  void on_accept(int client_fd, sockaddr_in addr);
  void setup_socket(int fd);

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;
  TransportConfig config_;

  int listen_fd_{-1};
  Endpoint local_endpoint_;

  std::unordered_map<uint64_t, std::shared_ptr<TcpConnection>> connections_;
  mutable std::mutex connections_mutex_;

  std::atomic<uint64_t> next_connection_id_{0};
  std::atomic<bool> running_{false};

  MessageCallback message_callback_;
  DisconnectCallback disconnect_callback_;
  ErrorCallback error_callback_;
};

/**
 * Shared Memory Transport for ultra-low latency intra-node communication
 * Achieves <10μs latency using lock-free ring buffers
 */
class SharedMemoryTransport : public Transport {
public:
  SharedMemoryTransport(std::shared_ptr<BufferPool> buffer_pool,
                        const std::string &name_prefix = "gossip_shm");
  ~SharedMemoryTransport() override;

  bool listen(const Endpoint &endpoint) override;
  std::shared_ptr<Connection> connect(const Endpoint &endpoint) override;
  void connect_async(const Endpoint &endpoint,
                     ConnectCallback callback) override;
  void stop() override;

  [[nodiscard]] TransportType type() const override {
    return TransportType::SHARED_MEMORY;
  }

  void set_message_callback(MessageCallback callback) override;
  void set_disconnect_callback(DisconnectCallback callback) override;
  void set_error_callback(ErrorCallback callback) override;

  [[nodiscard]] std::vector<std::shared_ptr<Connection>>
  connections() const override;
  [[nodiscard]] std::shared_ptr<Connection>
  get_connection(const Endpoint &endpoint) const override;

  /**
   * Poll for incoming messages (call from event loop)
   */
  void poll();

private:
  class ShmConnection;

  std::shared_ptr<BufferPool> buffer_pool_;
  std::string name_prefix_;

  std::unordered_map<uint64_t, std::shared_ptr<ShmConnection>> connections_;
  mutable std::mutex connections_mutex_;

  std::atomic<uint64_t> next_connection_id_{0};
  std::atomic<bool> running_{false};

  MessageCallback message_callback_;
  DisconnectCallback disconnect_callback_;
  ErrorCallback error_callback_;
};

/**
 * Factory for creating appropriate transport based on configuration
 */
class TransportFactory {
public:
  /**
   * Create a transport of the specified type
   */
  static std::unique_ptr<Transport>
  create(TransportType type, std::shared_ptr<EventLoop> event_loop,
         std::shared_ptr<BufferPool> buffer_pool,
         const TransportConfig &config = TransportConfig{});

  /**
   * Determine the best transport for communicating with a peer
   */
  static TransportType detect_best_transport(const Endpoint &local,
                                             const Endpoint &remote);
};

} // namespace gossip_rl::rpc
