# C++ RPC Framework

## Overview

The C++ RPC layer provides high-performance, zero-copy communication between agents. It achieves sub-10μs latency for intra-node communication using shared memory and sub-100μs for cross-node TCP.

## Architecture

```
┌─────────────────────────────────────────┐
│            Application Layer            │
│  (RPC Client / RPC Server / Services)   │
├─────────────────────────────────────────┤
│           Serialization Layer           │
│         (FlatBuffers, Zero-copy)        │
├─────────────────────────────────────────┤
│            Transport Layer              │
│    (TCP Transport / SHM Transport)      │
├─────────────────────────────────────────┤
│           Buffer Management             │
│    (BufferPool, Region Registration)    │
├─────────────────────────────────────────┤
│             Event Loop                  │
│    (io_uring on Linux / kqueue macOS)   │
└─────────────────────────────────────────┘
```

## Components

### Event Loop (`event_loop.hpp`)

Async I/O using io_uring (Linux) or kqueue (macOS):

```cpp
EventLoop::Config config;
config.ring_size = 256;
config.batch_size = 32;

auto loop = std::make_shared<EventLoop>(config);
loop->run();  // Start processing
```

**Operations supported:**
- `submit_read()` - Async read
- `submit_write()` - Async write
- `submit_accept()` - Accept connections
- `submit_connect()` - Initiate connection
- `submit_timeout()` - Timer events

### Buffer Pool (`buffer_pool.hpp`)

Zero-copy buffer management:

```cpp
BufferPoolConfig config;
config.buffer_size = 65536;
config.num_buffers = 256;

auto pool = std::make_shared<BufferPool>(config);

// Acquire buffer (returns to pool on destruction)
auto buffer = pool->acquire();
buffer->write(data, size);
```

**Features:**
- Pre-allocated buffer pool
- Registered with io_uring for zero-copy
- RAII-based automatic return

### Transport Layer

#### TCP Transport (`tcp_transport.cpp`)

```cpp
TcpTransport transport(event_loop, buffer_pool);

// Server
transport.listen(5000, [](auto conn) {
    // Handle new connection
});

// Client
transport.connect(endpoint, [](auto conn, bool success) {
    // Connection established
});
```

#### Shared Memory Transport (`shm_transport.cpp`)

For intra-node communication (~10μs latency):

```cpp
SharedMemoryTransport transport(event_loop, buffer_pool);
auto region = transport.create_region("gradient_shm", 1024*1024);
```

### RPC Server (`rpc_server.hpp`)

```cpp
auto service = std::make_shared<RpcService>();
service->name = "GossipService";

service->register_method(1, "exchange_gradients",
    [](const RpcContext& ctx, std::span<const uint8_t> payload) {
        // Handle gradient exchange
        RpcResponse response;
        response.status = RpcStatus::OK;
        return response;
    });

RpcServer server(event_loop, transport, buffer_pool);
server.register_service(service);
server.start(5000);
```

### RPC Client (`rpc_client.cpp`)

```cpp
RpcClient client(event_loop, transport, buffer_pool);

// Async call
auto future = client.call(endpoint, method_id, payload);
auto response = future.wait(std::chrono::seconds(5));

if (response.status == RpcStatus::OK) {
    // Process response
}
```

## Serialization (`serialization.hpp`)

Zero-copy tensor serialization:

```cpp
BufferWriter writer(buffer.get());

std::vector<uint64_t> dims = {1024, 512};
std::vector<float> data = ...;

serialize_tensor(writer, TensorDtype::FLOAT32, dims, data.data());
```

## Performance

| Metric | Value |
|--------|-------|
| Intra-node (SHM) | < 10μs |
| Cross-node (TCP) | < 100μs |
| Throughput | 1M+ msgs/sec |
| Zero-copy | Yes |

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Source Files

| File | Description |
|------|-------------|
| `cpp/include/rpc/event_loop.hpp` | Event loop interface |
| `cpp/include/rpc/buffer_pool.hpp` | Buffer management |
| `cpp/include/rpc/transport.hpp` | Transport abstraction |
| `cpp/include/rpc/rpc_server.hpp` | RPC server/client |
| `cpp/src/event_loop.cpp` | Event loop implementation |
| `cpp/src/tcp_transport.cpp` | TCP transport |
| `cpp/src/shm_transport.cpp` | Shared memory transport |
