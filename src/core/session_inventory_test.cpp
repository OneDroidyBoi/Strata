// Standalone CPU fixture: inventory and actual session-file persistence, no model/GPU.
#include "strata/core/session_inventory.hpp"
#include "strata/core/conversation_file.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
using namespace strata::core;
int main() {
    ConversationCache cache(1 << 20, 8);
    SavedConversation image;
    image.live.ids = {1, 2, 3}; image.live.gdn = {12, 13}; image.live.ple = {9};
    image.live.tails = {4}; image.live.dead = {5}; image.live.block_pos = {6};
    image.checkpoints.push_back(image.live);
    ConversationKv kv; kv.k.resize(4); kv.v.resize(4); image.kv.push_back(std::move(kv));
    assert(cache.put(std::move(image)));
    SessionInventory inventory("process-one");
    std::vector<SessionInventoryEntry> entries;
    std::string error;
    assert(inventory.collect(cache, true, true, 8, entries, error));
    assert(entries.size() == 2 && entries[0].id == "live" && entries[1].id == "parked-1");
    const auto generation = inventory.generation();
    const auto* parked = cache.find(1);
    const auto bytes_before = cache.bytes();
    const auto path = (std::filesystem::temp_directory_path() /
        ("strata-inventory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    size_t written = 0, read = 0;
    SessionFileIdentity identity{7, 8};
    SessionWriteOptions options; options.min_free_bytes = 0;
    assert(session_file_write(path, *parked, identity, written, error, options));
    SavedConversation restored;
    assert(session_file_read(path, identity, restored, read, error));
    assert(written == read && restored.live.ids == parked->live.ids);
    assert(restored.live.gdn == parked->live.gdn && restored.live.ple == parked->live.ple);
    assert(restored.live.tails == parked->live.tails && restored.live.dead == parked->live.dead);
    assert(restored.live.block_pos == parked->live.block_pos && restored.checkpoints.size() == 1);
    assert(cache.find(1) == parked && cache.bytes() == bytes_before && inventory.matches(generation));
    assert(inventory.collect(cache, true, true, 8, entries, error) && entries.size() == 2);
    inventory.invalidate(); assert(!inventory.matches(generation));
    assert(!SessionInventory("process-two").matches(generation));
    assert(!inventory.collect(cache, false, true, 8, entries, error) && entries.empty());
    assert(!inventory.collect(cache, true, false, 8, entries, error) && entries.empty());
    SavedConversation split; split.live.ids = {4}; split.stage_images.emplace_back();
    assert(cache.put(std::move(split)));
    assert(!inventory.collect(cache, true, true, 8, entries, error) && entries.empty());
    cache.take(0); // deque index shifts, entry identity must not.
    assert(cache.find(1) == nullptr && cache.find(2) != nullptr);
    assert(cache.index_of(2) == 0);
    cache.clear();
    assert(inventory.collect(cache, true, false, 0, entries, error) && entries.empty());
    // An idle cached slot is preserved even when the scheduler has no busy lease.
    std::vector<SessionInventorySlot> slots{{0, 8, 1, false, true, false, false, false, true},
                                          {1, 0, 0, false, false, false, false, false, true}};
    assert(session_inventory_append_slots(slots, entries, error));
    assert(entries.size() == 1 && entries[0].id == "slot-0" && entries[0].kind == "slot");
    for (int mode = 0; mode < 7; ++mode) {
        auto bad = slots;
        if (mode == 0) bad[0].active = true;
        if (mode == 1) bad[0].partial = true;
        if (mode == 2) bad[0].tainted = true;
        if (mode == 3) bad[0].images = true;
        if (mode == 4) bad[0].cached = false;
        if (mode == 5) bad[0].layout_valid = false;
        if (mode == 6) bad[1].checkpoints = 1;
        entries = {{"live", "live", 10}};
        assert(!session_inventory_append_slots(bad, entries, error) && entries.empty());
    }
    size_t target = 99;
    std::string restore_path;
    assert(session_slot_restore_parse("RESTORE_SLOT 1 /tmp/a path.bin", 2, target, restore_path, error));
    assert(target == 1 && restore_path == "/tmp/a path.bin");
    for (const auto& command : {"RESTORE_SLOT", "RESTORE_SLOT 0", "RESTORE_SLOT 0 ",
            "RESTORE_SLOT -1 x", "RESTORE_SLOT +1 x", "RESTORE_SLOT 01 x", "RESTORE_SLOT 2 x",
            "RESTORE_SLOT 0junk x", "RESTORE_SLOT 9999999999999999999999999 x"})
        assert(!session_slot_restore_parse(command, 2, target, restore_path, error));
    assert(!session_slot_restore_parse(std::string("RESTORE_SLOT 1 /tmp/a") + '\0' + "b", 2,
                                       target, restore_path, error));
    // Slot zero remains cached while an independent empty slot is restored.
    assert(session_slot_restore_ready(slots, 1, true, true, error));
    assert(!session_slot_restore_ready(slots, 0, true, true, error));
    assert(!session_slot_restore_ready(slots, 2, true, true, error));
    assert(!session_slot_restore_ready(slots, 1, false, true, error));
    assert(!session_slot_restore_ready(slots, 1, true, false, error));
    for (int mode = 0; mode < 7; ++mode) {
        auto busy = slots;
        if (mode == 0) busy[0].active = true;
        if (mode == 1) busy[0].partial = true;
        if (mode == 2) busy[1].tainted = true;
        if (mode == 3) busy[1].images = true;
        if (mode == 4) busy[1].checkpoints = 1;
        if (mode == 5) busy[1].layout_valid = false;
        if (mode == 6) busy[1].cached = true;
        assert(!session_slot_restore_ready(busy, 1, true, true, error));
    }
    auto quarantined = slots;
    quarantined[1].tainted = true; // partial GPU transfer cannot silently disappear from inventory.
    assert(!session_inventory_append_slots(quarantined, entries, error) && entries.empty());
    assert(slots[0].cached && slots[0].tokens == 8 && !slots[1].tainted);
    assert(session_slot_release_parse("RELEASE_SLOT 0", 2, target, error) && target == 0);
    for (const auto& command : {"RELEASE_SLOT", "RELEASE_SLOT ", "RELEASE_SLOT -1", "RELEASE_SLOT +1",
            "RELEASE_SLOT 01", "RELEASE_SLOT 2", "RELEASE_SLOT 0 junk", "RELEASE_SLOT 0 ",
            "RELEASE_SLOT 99999999999999999999999"})
        assert(!session_slot_release_parse(command, 2, target, error));
    assert(session_slot_release_ready(slots, 0, true, error));
    assert(!session_slot_release_ready(slots, 1, true, error)); // empty is never silently released
    assert(!session_slot_release_ready(slots, 0, false, error));
    for (int mode = 0; mode < 7; ++mode) {
        auto blocked = slots;
        if (mode == 0) blocked[1].active = true;
        if (mode == 1) blocked[1].partial = true;
        if (mode == 2) blocked[0].tainted = true;
        if (mode == 3) blocked[0].images = true;
        if (mode == 4) blocked[0].cached = false;
        if (mode == 5) blocked[0].layout_valid = false;
        if (mode == 6) blocked[0].tokens = 0;
        assert(!session_slot_release_ready(blocked, 0, true, error));
    }
    struct Metadata { std::vector<int> ids; bool cached = false; std::chrono::steady_clock::time_point start{}; };
    std::vector<Metadata> metadata{{{1,2,3},true,{}},{{7,8},true,{}}};
    assert(!session_slot_release(metadata, slots, 0, false, error));
    assert(metadata[0].ids.size() == 3 && metadata[1].ids.size() == 2);
    assert(session_slot_release(metadata, slots, 0, true, error));
    assert(metadata[0].ids.empty() && !metadata[0].cached);
    assert(metadata[1].ids == std::vector<int>({7,8}) && metadata[1].cached);
    const std::vector<int32_t> current{1,2,3};
    const std::vector<int64_t> requested{1,2,3};
    assert(session_owned_prefix_valid(false, false, true, 1, current, requested)); // max_new=1 / first EOS
    assert(!session_owned_prefix_valid(true, false, true, 1, current, requested)); // cancel even at exact boundary
    assert(!session_owned_prefix_valid(false, true, true, 1, current, requested));
    assert(!session_owned_prefix_valid(false, false, false, 1, current, requested));
    assert(!session_owned_prefix_valid(false, false, true, 0, current, requested));
    assert(!session_owned_prefix_valid(false, false, true, 1, std::vector<int32_t>{1,2}, requested));
    assert(!session_owned_prefix_valid(false, false, true, 1, std::vector<int32_t>{1,2,4}, requested));
    assert(!session_owned_prefix_valid(false, false, true, 1, std::vector<int32_t>{}, std::vector<int64_t>{}));
    std::filesystem::remove(path);
    std::cout << "session inventory tests passed\n";
}
