# C++ Tests
enable_testing()

# Test sources
add_executable(test_buffer_pool test_buffer_pool.cpp)
target_link_libraries(test_buffer_pool
    gossip_rpc
    GTest::gtest_main
)

add_executable(test_event_loop test_event_loop.cpp)
target_link_libraries(test_event_loop
    gossip_rpc
    GTest::gtest_main
)

add_executable(test_transport test_transport.cpp)
target_link_libraries(test_transport
    gossip_rpc
    GTest::gtest_main
)

add_executable(test_rpc test_rpc.cpp)
target_link_libraries(test_rpc
    gossip_rpc
    GTest::gtest_main
)

# Add tests
add_test(NAME BufferPoolTest COMMAND test_buffer_pool)
add_test(NAME EventLoopTest COMMAND test_event_loop)
add_test(NAME TransportTest COMMAND test_transport)
add_test(NAME RpcTest COMMAND test_rpc)
