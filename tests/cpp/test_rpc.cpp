#include "rpc/rpc_server.hpp"
#include "rpc/serialization.hpp"
#include <gtest/gtest.h>

using namespace gossip_rl::rpc;

namespace gossip_rl::rpc {

class RpcTest : public ::testing::Test {
protected:
  void SetUp() override {
    EventLoop::Config ev_config;
    event_loop_ = std::make_shared<EventLoop>(ev_config);

    BufferPoolConfig bp_config;
    buffer_pool_ = std::make_shared<BufferPool>(bp_config);
  }

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;

  void complete_future(RpcFuture &future) {
    std::lock_guard lock(future.state_->mutex);
    future.state_->response.status = RpcStatus::OK;
    future.state_->completed = true;
    future.state_->cv.notify_all();
  }
};

TEST_F(RpcTest, RpcHeaderSize) {
  EXPECT_EQ(sizeof(RpcHeader), 32);
  EXPECT_EQ(RpcHeader::SIZE, 32);
}

TEST_F(RpcTest, RpcHeaderPacking) {
  RpcHeader header;
  header.request_id = 12345;
  header.method_id = 42;
  header.payload_size = 1024;
  header.flags = RpcFlags::COMPRESSED;

  EXPECT_EQ(header.magic, 0x47525043); // "GRPC"
  EXPECT_EQ(header.version, 1);
}

TEST_F(RpcTest, ServiceRegistration) {
  auto service = std::make_shared<RpcService>();
  service->name = "TestService";

  service->register_method(
      1, "echo", [](const RpcContext &ctx, std::span<const uint8_t> payload) {
        (void)ctx;
        (void)payload;
        RpcResponse response;
        response.status = RpcStatus::OK;
        return response;
      });

  EXPECT_EQ(service->methods.size(), 1);
  EXPECT_TRUE(service->methods.count(1) > 0);
}

TEST_F(RpcTest, BufferWriterReader) {
  auto buffer = buffer_pool_->acquire();
  BufferWriter writer(buffer.get());

  // Write some data
  uint32_t value = 42;
  EXPECT_TRUE(writer.write_value(value));

  std::string str = "hello";
  EXPECT_TRUE(writer.write_string(str));

  // Read it back
  BufferReader reader(*buffer);

  uint32_t read_value;
  EXPECT_TRUE(reader.read_value(read_value));
  EXPECT_EQ(read_value, 42);

  std::string read_str;
  EXPECT_TRUE(reader.read_string(read_str));
  EXPECT_EQ(read_str, "hello");
}

TEST_F(RpcTest, TensorSerialization) {
  auto buffer = buffer_pool_->acquire();
  BufferWriter writer(buffer.get());

  std::vector<uint64_t> dims = {2, 3};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  EXPECT_TRUE(
      serialize_tensor(writer, TensorDtype::FLOAT32, dims, data.data()));

  EXPECT_GT(buffer->size(), 0);
}

TEST_F(RpcTest, RpcFuture) {
  RpcFuture future;

  EXPECT_FALSE(future.is_ready());

  // Complete in another thread
  std::thread completer([this, &future]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    complete_future(future);
  });

  auto response = future.wait(std::chrono::milliseconds(1000));
  completer.join();

  EXPECT_EQ(response.status, RpcStatus::OK);
}

} // namespace gossip_rl::rpc

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
