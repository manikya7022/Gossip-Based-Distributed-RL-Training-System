#pragma once

#define GOSSIP_RL_VERSION_MAJOR 1
#define GOSSIP_RL_VERSION_MINOR 0
#define GOSSIP_RL_VERSION_PATCH 0
#define GOSSIP_RL_VERSION "1.0.0"

namespace gossip_rl {

constexpr int kVersionMajor = GOSSIP_RL_VERSION_MAJOR;
constexpr int kVersionMinor = GOSSIP_RL_VERSION_MINOR;
constexpr int kVersionPatch = GOSSIP_RL_VERSION_PATCH;

inline const char* GetVersion() {
    return GOSSIP_RL_VERSION;
}

}  // namespace gossip_rl
