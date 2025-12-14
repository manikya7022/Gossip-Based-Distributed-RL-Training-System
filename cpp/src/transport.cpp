#include "rpc/transport.hpp"

namespace gossip_rl::rpc {

std::unique_ptr<Transport> TransportFactory::create(
    TransportType type, std::shared_ptr<EventLoop> event_loop,
    std::shared_ptr<BufferPool> buffer_pool, const TransportConfig &config) {

  switch (type) {
  case TransportType::TCP:
    return std::make_unique<TcpTransport>(std::move(event_loop),
                                          std::move(buffer_pool), config);

  case TransportType::SHARED_MEMORY:
    return std::make_unique<SharedMemoryTransport>(std::move(buffer_pool),
                                                   "gossip_shm");

  case TransportType::AUTO:
    // Default to TCP for now
    return std::make_unique<TcpTransport>(std::move(event_loop),
                                          std::move(buffer_pool), config);

  default:
    spdlog::error("Unknown transport type");
    return nullptr;
  }
}

TransportType TransportFactory::detect_best_transport(const Endpoint &local,
                                                      const Endpoint &remote) {

  // If same host, prefer shared memory
  if (local.host == remote.host || remote.host == "localhost" ||
      remote.host == "127.0.0.1" || remote.host == "::1") {
    return TransportType::SHARED_MEMORY;
  }

  // Otherwise use TCP
  return TransportType::TCP;
}

} // namespace gossip_rl::rpc
