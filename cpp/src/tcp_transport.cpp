#include "rpc/transport.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>

namespace gossip_rl::rpc {

/**
 * TCP Connection implementation
 */
class TcpTransport::TcpConnection : public Connection {
public:
  TcpConnection(TcpTransport *transport, int fd, const Endpoint &local,
                const Endpoint &remote, uint64_t id)
      : transport_(transport), fd_(fd), local_endpoint_(local),
        remote_endpoint_(remote), id_(id), state_(ConnectionState::CONNECTED) {
    stats_.connected_at = std::chrono::steady_clock::now();
    stats_.last_activity = stats_.connected_at;
  }

  ~TcpConnection() override { close(); }

  bool send(std::unique_ptr<Buffer> buffer) override {
    return send(std::move(buffer), nullptr);
  }

  bool send(std::unique_ptr<Buffer> buffer,
            std::function<void(bool)> callback) override {
    if (state_ != ConnectionState::CONNECTED) {
      if (callback)
        callback(false);
      return false;
    }

    auto *event_loop = transport_->event_loop_.get();
    auto *buf_ptr = buffer.get();
    auto buf_data = buf_ptr->readable_span();

    // Use shared_ptr with custom deleter to make lambda copyable and return
    // buffer to pool
    std::shared_ptr<Buffer> shared_buffer(
        buffer.release(), [pool = transport_->buffer_pool_](Buffer *b) {
          pool->release(std::unique_ptr<Buffer>(b));
        });

    event_loop->submit_send(
        fd_, buf_data.data(), buf_data.size(), 0,
        [this, shared_buffer, callback](int result, void *) {
          if (result >= 0) {
            stats_.bytes_sent.fetch_add(result, std::memory_order_relaxed);
            stats_.messages_sent.fetch_add(1, std::memory_order_relaxed);
            stats_.last_activity = std::chrono::steady_clock::now();
            if (callback)
              callback(true);
          } else {
            stats_.errors.fetch_add(1, std::memory_order_relaxed);
            if (callback)
              callback(false);
          }
        },
        nullptr);

    return true;
  }

  void close() override {
    ConnectionState expected = ConnectionState::CONNECTED;
    if (state_.compare_exchange_strong(expected, ConnectionState::CLOSING)) {
      if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
      }
      state_.store(ConnectionState::CLOSED);
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
    return TransportType::TCP;
  }

  [[nodiscard]] const ConnectionStats &stats() const override { return stats_; }

  [[nodiscard]] bool is_healthy() const override {
    return state_.load(std::memory_order_acquire) == ConnectionState::CONNECTED;
  }

  [[nodiscard]] uint64_t id() const override { return id_; }

  void set_user_data(void *data) override { user_data_ = data; }

  [[nodiscard]] void *user_data() const override { return user_data_; }

  void start_recv();

private:
  TcpTransport *transport_;
  int fd_;
  Endpoint local_endpoint_;
  Endpoint remote_endpoint_;
  uint64_t id_;
  std::atomic<ConnectionState> state_;
  ConnectionStats stats_;
  void *user_data_{nullptr};
  std::unique_ptr<Buffer> recv_buffer_;
};

void TcpTransport::TcpConnection::start_recv() {
  if (state_.load() != ConnectionState::CONNECTED)
    return;

  recv_buffer_ = transport_->buffer_pool_->acquire();
  if (!recv_buffer_) {
    spdlog::error("Failed to acquire recv buffer");
    return;
  }

  auto *event_loop = transport_->event_loop_.get();
  event_loop->submit_recv(
      fd_, recv_buffer_->data(), recv_buffer_->capacity(), 0,
      [this](int result, void *) {
        if (result > 0) {
          recv_buffer_->set_size(result);
          stats_.bytes_received.fetch_add(result, std::memory_order_relaxed);
          stats_.messages_received.fetch_add(1, std::memory_order_relaxed);
          stats_.last_activity = std::chrono::steady_clock::now();

          if (transport_->message_callback_) {
            transport_->message_callback_(this, std::move(recv_buffer_));
          }

          // Continue receiving
          start_recv();
        } else if (result == 0) {
          // Connection closed by peer
          if (transport_->disconnect_callback_) {
            transport_->disconnect_callback_(this);
          }
          close();
        } else {
          // Error
          stats_.errors.fetch_add(1, std::memory_order_relaxed);
          if (transport_->error_callback_) {
            transport_->error_callback_(this, strerror(-result));
          }
          close();
        }
      },
      nullptr);
}

// TcpTransport implementation

TcpTransport::TcpTransport(std::shared_ptr<EventLoop> event_loop,
                           std::shared_ptr<BufferPool> buffer_pool,
                           const TransportConfig &config)
    : event_loop_(std::move(event_loop)), buffer_pool_(std::move(buffer_pool)),
      config_(config) {}

TcpTransport::~TcpTransport() { stop(); }

void TcpTransport::setup_socket(int fd) {
  // Set non-blocking
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  // TCP options
  if (config_.tcp_nodelay) {
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
  }

  if (config_.tcp_keepalive) {
    int flag = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));

#ifdef __linux__
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &config_.tcp_keepalive_idle,
               sizeof(config_.tcp_keepalive_idle));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &config_.tcp_keepalive_interval,
               sizeof(config_.tcp_keepalive_interval));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &config_.tcp_keepalive_count,
               sizeof(config_.tcp_keepalive_count));
#endif
  }

  // Buffer sizes
  int recv_size = static_cast<int>(config_.recv_buffer_size);
  int send_size = static_cast<int>(config_.send_buffer_size);
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &recv_size, sizeof(recv_size));
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size));
}

bool TcpTransport::listen(const Endpoint &endpoint) {
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    spdlog::error("Failed to create socket: {}", strerror(errno));
    return false;
  }

  if (config_.reuse_addr) {
    int flag = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
  }

  if (config_.reuse_port) {
    int flag = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &flag, sizeof(flag));
  }

  setup_socket(listen_fd_);

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(endpoint.port);

  if (endpoint.host.empty() || endpoint.host == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else {
    inet_pton(AF_INET, endpoint.host.c_str(), &addr.sin_addr);
  }

  if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    spdlog::error("Failed to bind: {}", strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  if (::listen(listen_fd_, SOMAXCONN) < 0) {
    spdlog::error("Failed to listen: {}", strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  local_endpoint_ = endpoint;
  running_.store(true);

  // Start accepting connections
  auto accept_callback = [this](int result, void *) {
    if (result >= 0) {
      sockaddr_in client_addr{};
      socklen_t len = sizeof(client_addr);
      getpeername(result, reinterpret_cast<sockaddr *>(&client_addr), &len);
      on_accept(result, client_addr);
    }

    // Continue accepting
    if (running_.load()) {
      event_loop_->submit_accept(
          listen_fd_, nullptr, nullptr,
          [this](int result, void *) {
            if (result >= 0 && running_.load()) {
              sockaddr_in client_addr{};
              socklen_t len = sizeof(client_addr);
              getpeername(result, reinterpret_cast<sockaddr *>(&client_addr),
                          &len);
              on_accept(result, client_addr);
            }
          },
          nullptr);
    }
  };

  event_loop_->submit_accept(listen_fd_, nullptr, nullptr, accept_callback,
                             nullptr);

  spdlog::info("TCP transport listening on {}:{}", endpoint.host,
               endpoint.port);
  return true;
}

void TcpTransport::on_accept(int client_fd, sockaddr_in addr) {
  setup_socket(client_fd);

  char ip_str[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));

  Endpoint remote{ip_str, ntohs(addr.sin_port)};

  uint64_t conn_id = next_connection_id_.fetch_add(1);
  auto conn = std::make_shared<TcpConnection>(this, client_fd, local_endpoint_,
                                              remote, conn_id);

  {
    std::lock_guard lock(connections_mutex_);
    connections_[conn_id] = conn;
  }

  conn->start_recv();

  spdlog::debug("Accepted connection from {}:{}", ip_str, ntohs(addr.sin_port));
}

std::shared_ptr<Connection> TcpTransport::connect(const Endpoint &endpoint) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    spdlog::error("Failed to create socket: {}", strerror(errno));
    return nullptr;
  }

  setup_socket(fd);

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(endpoint.port);
  inet_pton(AF_INET, endpoint.host.c_str(), &addr.sin_addr);

  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    if (errno != EINPROGRESS) {
      spdlog::error("Failed to connect: {}", strerror(errno));
      ::close(fd);
      return nullptr;
    }
  }

  // Get local endpoint
  sockaddr_in local_addr{};
  socklen_t len = sizeof(local_addr);
  getsockname(fd, reinterpret_cast<sockaddr *>(&local_addr), &len);

  char local_ip[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &local_addr.sin_addr, local_ip, sizeof(local_ip));
  Endpoint local{local_ip, ntohs(local_addr.sin_port)};

  uint64_t conn_id = next_connection_id_.fetch_add(1);
  auto conn =
      std::make_shared<TcpConnection>(this, fd, local, endpoint, conn_id);

  {
    std::lock_guard lock(connections_mutex_);
    connections_[conn_id] = conn;
  }

  conn->start_recv();

  spdlog::debug("Connected to {}:{}", endpoint.host, endpoint.port);
  return conn;
}

void TcpTransport::connect_async(const Endpoint &endpoint,
                                 ConnectCallback callback) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    if (callback)
      callback(nullptr, false);
    return;
  }

  setup_socket(fd);

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(endpoint.port);
  inet_pton(AF_INET, endpoint.host.c_str(), &addr.sin_addr);

  event_loop_->submit_connect(
      fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr),
      [this, fd, endpoint, callback](int result, void *) {
        if (result < 0) {
          ::close(fd);
          if (callback)
            callback(nullptr, false);
          return;
        }

        sockaddr_in local_addr{};
        socklen_t len = sizeof(local_addr);
        getsockname(fd, reinterpret_cast<sockaddr *>(&local_addr), &len);

        char local_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &local_addr.sin_addr, local_ip, sizeof(local_ip));
        Endpoint local{local_ip, ntohs(local_addr.sin_port)};

        uint64_t conn_id = next_connection_id_.fetch_add(1);
        auto conn =
            std::make_shared<TcpConnection>(this, fd, local, endpoint, conn_id);

        {
          std::lock_guard lock(connections_mutex_);
          connections_[conn_id] = conn;
        }

        conn->start_recv();

        if (callback)
          callback(conn.get(), true);
      },
      nullptr);
}

void TcpTransport::stop() {
  running_.store(false);

  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }

  std::lock_guard lock(connections_mutex_);
  for (auto &[id, conn] : connections_) {
    conn->close();
  }
  connections_.clear();
}

void TcpTransport::set_message_callback(MessageCallback callback) {
  message_callback_ = std::move(callback);
}

void TcpTransport::set_disconnect_callback(DisconnectCallback callback) {
  disconnect_callback_ = std::move(callback);
}

void TcpTransport::set_error_callback(ErrorCallback callback) {
  error_callback_ = std::move(callback);
}

std::vector<std::shared_ptr<Connection>> TcpTransport::connections() const {
  std::lock_guard lock(connections_mutex_);
  std::vector<std::shared_ptr<Connection>> result;
  result.reserve(connections_.size());
  for (const auto &[id, conn] : connections_) {
    result.push_back(conn);
  }
  return result;
}

std::shared_ptr<Connection>
TcpTransport::get_connection(const Endpoint &endpoint) const {
  std::lock_guard lock(connections_mutex_);
  for (const auto &[id, conn] : connections_) {
    if (conn->remote_endpoint() == endpoint) {
      return conn;
    }
  }
  return nullptr;
}

} // namespace gossip_rl::rpc
