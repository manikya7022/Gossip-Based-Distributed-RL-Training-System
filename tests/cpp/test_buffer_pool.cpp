#include "rpc/buffer_pool.hpp"
#include <gtest/gtest.h>

using namespace gossip_rl::rpc;

class BufferPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_.buffer_size = 4096;
    config_.initial_count = 10;
    config_.max_count = 100;
  }

  BufferPoolConfig config_;
};

TEST_F(BufferPoolTest, CreatePool) {
  BufferPool pool(config_);

  EXPECT_EQ(pool.pool_size(), config_.initial_count);
  EXPECT_EQ(pool.available(), config_.initial_count);
}

TEST_F(BufferPoolTest, AcquireBuffer) {
  BufferPool pool(config_);

  auto buffer = pool.acquire();
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(buffer->capacity(), config_.buffer_size);
  EXPECT_EQ(buffer->size(), 0);
  EXPECT_EQ(pool.available(), config_.initial_count - 1);
}

TEST_F(BufferPoolTest, ReleaseBuffer) {
  BufferPool pool(config_);

  auto buffer = pool.acquire();
  ASSERT_NE(buffer, nullptr);

  size_t available_before = pool.available();
  pool.release(std::move(buffer));

  EXPECT_EQ(pool.available(), available_before + 1);
}

TEST_F(BufferPoolTest, BufferWriteRead) {
  BufferPool pool(config_);
  auto buffer = pool.acquire();

  const char *data = "Hello, World!";
  size_t len = strlen(data);

  size_t written = buffer->write(data, len);
  EXPECT_EQ(written, len);
  EXPECT_EQ(buffer->size(), len);

  char read_data[32] = {0};
  size_t read = buffer->read(read_data, len);
  EXPECT_EQ(read, len);
  EXPECT_STREQ(read_data, data);
}

TEST_F(BufferPoolTest, PoolExpansion) {
  BufferPoolConfig small_config;
  small_config.buffer_size = 1024;
  small_config.initial_count = 2;
  small_config.max_count = 10;

  BufferPool pool(small_config);

  std::vector<std::unique_ptr<Buffer>> buffers;

  // Acquire more than initial count
  for (int i = 0; i < 5; ++i) {
    auto buf = pool.acquire();
    ASSERT_NE(buf, nullptr);
    buffers.push_back(std::move(buf));
  }

  // Pool should have expanded
  EXPECT_GT(pool.pool_size(), small_config.initial_count);
}

TEST_F(BufferPoolTest, PoolStats) {
  BufferPool pool(config_);

  auto buf1 = pool.acquire();
  auto buf2 = pool.acquire();

  const auto &stats = pool.stats();
  EXPECT_EQ(stats.allocations.load(), 2);
  EXPECT_EQ(stats.current_usage.load(), 2);

  pool.release(std::move(buf1));

  EXPECT_EQ(stats.deallocations.load(), 1);
  EXPECT_EQ(stats.current_usage.load(), 1);
}

#ifdef __linux__
TEST_F(BufferPoolTest, SharedMemoryRegion) {
  auto region = SharedMemoryRegion::create("test_region", 4096);
  ASSERT_NE(region, nullptr);

  EXPECT_EQ(region->size(), 4096);
  EXPECT_NE(region->data(), nullptr);

  // Write and read
  memcpy(region->data(), "test", 5);
  EXPECT_STREQ(static_cast<char *>(region->data()), "test");

  // Open existing
  auto region2 = SharedMemoryRegion::open("test_region");
  ASSERT_NE(region2, nullptr);
  EXPECT_STREQ(static_cast<char *>(region2->data()), "test");
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
