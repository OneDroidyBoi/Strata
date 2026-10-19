// CPU-only, read-only inventory for opt-in graceful shutdown. No token text is exposed.
#pragma once
#include "strata/core/conversation_cache.hpp"
#include <limits>
#include <charconv>
#include <string>
#include <vector>
namespace strata::core {
struct SessionInventoryEntry { std::string id, kind; size_t tokens = 0; };
// Every allocated slot is inspected, including ones the scheduler does not call busy.
struct SessionInventorySlot {
    size_t index = 0, tokens = 0, checkpoints = 0;
    bool active = false, cached = false, partial = false, tainted = false, images = false;
    bool layout_valid = true;
};
inline bool session_inventory_append_slots(const std::vector<SessionInventorySlot>& slots,
                                          std::vector<SessionInventoryEntry>& entries, std::string& error) {
    for (const auto& slot : slots) {
        const bool empty = !slot.tokens && !slot.checkpoints;
        if (slot.active || slot.partial || slot.tainted || slot.images || !slot.layout_valid ||
            (!empty && (!slot.cached || !slot.tokens)) || (empty && slot.cached)) {
            entries.clear(); error = "batch slot " + std::to_string(slot.index) +
                " is active, partial, tainted, or cannot be exported completely"; return false;
        }
    }
    for (const auto& slot : slots)
        if (slot.tokens) entries.push_back({"slot-" + std::to_string(slot.index), "slot", slot.tokens});
    return true;
}
// Parse the whole slot selector, never an atoi prefix; keep spaces in the filesystem path.
inline bool session_slot_restore_parse(const std::string& command, size_t count, size_t& slot,
                                       std::string& path, std::string& error) {
    constexpr size_t prefix = 13; // "RESTORE_SLOT "
    const size_t end = command.find(' ', prefix);
    if (command.rfind("RESTORE_SLOT ", 0) != 0 || end == std::string::npos || end + 1 == command.size()) {
        error = "expected RESTORE_SLOT <slot> <path>"; return false;
    }
    size_t parsed = 0;
    const auto result = std::from_chars(command.data() + prefix, command.data() + end, parsed);
    if (result.ec != std::errc() || result.ptr != command.data() + end || parsed >= count ||
        command.substr(prefix, end - prefix) != std::to_string(parsed)) {
        error = "invalid restore slot"; return false;
    }
    const std::string candidate = command.substr(end + 1);
    if (candidate.find('\0') != std::string::npos || candidate.find_first_of("\r\n") != std::string::npos) {
        error = "restore path contains a protocol control character"; return false;
    }
    slot = parsed; path = candidate; return true;
}
inline bool session_slot_restore_ready(const std::vector<SessionInventorySlot>& slots, size_t target,
                                       bool supported, bool runtime_empty, std::string& error) {
    if (!supported || target >= slots.size()) { error = "batch slot restore is not supported"; return false; }
    for (const auto& slot : slots)
        if (slot.active || slot.partial) { error = "batch must be fully quiescent before slot restore"; return false; }
    const auto& slot = slots[target];
    if (!runtime_empty || !slot.layout_valid || slot.tokens || slot.checkpoints || slot.cached || slot.tainted || slot.images) {
        error = "restore target must be an empty, untainted slot"; return false;
    }
    return true;
}
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
