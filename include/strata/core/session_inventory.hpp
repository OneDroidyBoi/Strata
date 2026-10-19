// CPU-only, read-only inventory for opt-in graceful shutdown. No token text is exposed.
#pragma once
#include "strata/core/conversation_cache.hpp"
#include <limits>
#include <string>
#include <vector>
namespace strata::core {
struct SessionInventoryEntry { std::string id, kind; size_t tokens = 0; };
class SessionInventory {
    std::string nonce_;
    uint64_t epoch_ = 0;
    bool exhausted_ = false;
public:
    explicit SessionInventory(std::string nonce) : nonce_(std::move(nonce)) {}
    void invalidate() {
        if (epoch_ == std::numeric_limits<uint64_t>::max()) exhausted_ = true;
        else ++epoch_;
    }
    std::string generation() const { return nonce_ + "-" + std::to_string(epoch_); }
    bool matches(const std::string& generation) const { return !exhausted_ && generation == this->generation(); }
    bool collect(const ConversationCache& cache, bool supported, bool live_ok, size_t live_tokens,
                 std::vector<SessionInventoryEntry>& out, std::string& error) const {
        out.clear();
        if (!supported || exhausted_ || (!live_ok && live_tokens)) {
            error = "session inventory cannot represent this layout or incomplete live state"; return false;
        }
        bool complete = true;
        cache.visit([&](uint64_t, const SavedConversation& image) {
            if (image.live.ids.empty() || !image.stage_images.empty() || !image.live.stage_parts.empty()) complete = false;
            for (const auto& cp : image.checkpoints) if (!cp.stage_parts.empty()) complete = false;
        });
        if (!complete) { error = "session inventory cannot represent a parked layout"; return false; }
        if (live_ok && live_tokens) out.push_back({"live", "live", live_tokens});
        cache.visit([&](uint64_t serial, const SavedConversation& image) {
            out.push_back({"parked-" + std::to_string(serial), "parked", image.live.ids.size()});
        });
        return true;
    }
};
} // namespace strata::core
