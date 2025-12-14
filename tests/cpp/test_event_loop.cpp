#include "rpc/event_loop.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using namespace gossip_rl::rpc;

class EventLoopTest : public ::testing::Test {
protected:
  void SetUp() override {
    EventLoop::Config config;
    config.ring_size = 256;
    config.idle_timeout = std::chrono::milliseconds(10);
    loop_ = std::make_shared<EventLoop>(config);
  }

  std::shared_ptr<EventLoop> loop_;
};

TEST_F(EventLoopTest, Create) {
  EXPECT_FALSE(loop_->is_running());
  EXPECT_EQ(loop_->pending_count(), 0);
}

TEST_F(EventLoopTest, RunOnce) {
  int result = loop_->run_once();
  // Should return 0 with no operations
  EXPECT_EQ(result, 0);
}

TEST_F(EventLoopTest, StartStop) {
  std::thread runner([this] { loop_->run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_TRUE(loop_->is_running());

  loop_->stop();
  runner.join();

  EXPECT_FALSE(loop_->is_running());
}

#ifdef USE_IO_URING
TEST_F(EventLoopTest, TimeoutOperation) {
  bool completed = false;

  loop_->submit_timeout(
      std::chrono::milliseconds(10),
      [&completed](int result, void *) { completed = true; }, nullptr);

  // Run until completion
  for (int i = 0; i < 10 && !completed; ++i) {
    loop_->run_once();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  EXPECT_TRUE(completed);
}

TEST_F(EventLoopTest, Stats) {
  const auto &stats = loop_->stats();

  EXPECT_EQ(stats.total_operations.load(), 0);
  EXPECT_EQ(stats.completed_operations.load(), 0);
}
#endif

TEST_F(EventLoopTest, EventLoopThread) {
  {
    EventLoopThread thread(loop_);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(loop_->is_running());
  }

  // Should be stopped after destruction
  EXPECT_FALSE(loop_->is_running());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
