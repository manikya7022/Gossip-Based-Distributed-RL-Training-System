# C++ RPC Library
add_library(gossip_rpc SHARED
    src/event_loop.cpp
    src/buffer_pool.cpp
    src/transport.cpp
    src/tcp_transport.cpp
    src/shm_transport.cpp
    src/rpc_server.cpp
    src/rpc_client.cpp
)

target_include_directories(gossip_rpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gossip_rpc PUBLIC
    Threads::Threads
    spdlog::spdlog
    flatbuffers
)

if(PLATFORM_LINUX AND USE_URING)
    if(URING_LIB)
        target_link_libraries(gossip_rpc PUBLIC ${URING_LIB})
    else()
        target_link_libraries(gossip_rpc PUBLIC uring)
    endif()
endif()

# Set compile features
target_compile_features(gossip_rpc PUBLIC cxx_std_20)

# Platform-specific settings
if(PLATFORM_MACOS)
    target_compile_definitions(gossip_rpc PRIVATE USE_KQUEUE)
elseif(PLATFORM_LINUX)
    if(USE_URING)
        target_compile_definitions(gossip_rpc PRIVATE USE_IO_URING)
    else()
        target_compile_definitions(gossip_rpc PRIVATE USE_EPOLL)
    endif()
endif()

# Install
install(TARGETS gossip_rpc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
