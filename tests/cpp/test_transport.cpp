#include "rpc/transport.hpp"
#include <gtest/gtest.h>

using namespace gossip_rl::rpc;

class TransportTest : public ::testing::Test {
protected:
  void SetUp() override {
    EventLoop::Config ev_config;
    ev_config.ring_size = 256;
    event_loop_ = std::make_shared<EventLoop>(ev_config);

    BufferPoolConfig bp_config;
    bp_config.initial_count = 10;
    buffer_pool_ = std::make_shared<BufferPool>(bp_config);
  }

  std::shared_ptr<EventLoop> event_loop_;
  std::shared_ptr<BufferPool> buffer_pool_;
};

TEST_F(TransportTest, EndpointToString) {
  Endpoint ep{"localhost", 8080};
  EXPECT_EQ(ep.to_string(), "localhost:8080");
}

TEST_F(TransportTest, EndpointEquality) {
  Endpoint ep1{"localhost", 8080};
  Endpoint ep2{"localhost", 8080};
  Endpoint ep3{"localhost", 8081};

  EXPECT_EQ(ep1, ep2);
  EXPECT_NE(ep1, ep3);
}

TEST_F(TransportTest, TransportFactoryTcp) {
  auto transport =
      TransportFactory::create(TransportType::TCP, event_loop_, buffer_pool_);

  ASSERT_NE(transport, nullptr);
  EXPECT_EQ(transport->type(), TransportType::TCP);
}

TEST_F(TransportTest, TransportFactoryShm) {
  auto transport = TransportFactory::create(TransportType::SHARED_MEMORY,
                                            event_loop_, buffer_pool_);

  ASSERT_NE(transport, nullptr);
  EXPECT_EQ(transport->type(), TransportType::SHARED_MEMORY);
}

TEST_F(TransportTest, DetectBestTransport) {
  Endpoint local{"localhost", 8080};
  Endpoint remote_local{"localhost", 8081};
  Endpoint remote_external{"192.168.1.100", 8080};

  // Same host should prefer shared memory
  auto type1 = TransportFactory::detect_best_transport(local, remote_local);
  EXPECT_EQ(type1, TransportType::SHARED_MEMORY);

  // Different host should use TCP
  auto type2 = TransportFactory::detect_best_transport(local, remote_external);
  EXPECT_EQ(type2, TransportType::TCP);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
