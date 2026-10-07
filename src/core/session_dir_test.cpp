// CPU-only tests for --session-dir (include/strata/core/session_dir.hpp): write-behind, supersede, claim, caps,
// fetch, preload, foreign files.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/session_dir.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint head(const std::vector<int32_t>& ids, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn = bytes_of(513, seed); c.ple = bytes_of(17, seed); c.tails = bytes_of(33, seed);
    c.dead = bytes_of(8, seed); c.block_pos = bytes_of(8, seed);
    c.used = seed;
    return c;
}

std::vector<int32_t> tokens(int32_t base, size_t n) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + int32_t(i);
    return v;
}

// a conversation: `ids` live, checkpoints at 1/4 and 1/2 of it, one small K/V layer; `stages` later stage images
SavedConversation conv(const std::vector<int32_t>& ids, uint8_t seed, int stages = 0) {
    auto one = [&](int64_t lo, int64_t hi, uint8_t s) {
        SavedConversation c;
        for (size_t i = 0; i < c.geometry.size(); ++i) c.geometry[i] = int64_t(7 + i);
        c.layer_lo = lo; c.layer_hi = hi;
        c.cvec = false;   // (the default is true)
        c.live = head(ids, s);
        c.checkpoints = {head({ids.begin(), ids.begin() + ids.size() / 4}, s + 1),
                         head({ids.begin(), ids.begin() + ids.size() / 2}, s + 2)};
        ConversationKv kv;
        kv.format = 2; kv.cells = 64; kv.heads = 1; kv.head_dim = 8; kv.page_size = 64; kv.idx_dim = 4;
        kv.k.resize(1000 + s); kv.v.resize(900 + s);
        c.kv.push_back(std::move(kv));
        return c;
    };
    SavedConversation c = one(0, stages ? 10 : 48, seed);
    for (int k = 0; k < stages; ++k) c.stage_images.push_back(one(10 + 12 * k, 22 + 12 * k, uint8_t(seed + 10 * (k + 1))));
    return c;
}

size_t session_files(const fs::path& d) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(d)) n += e.path().extension() == ".session";
    return n;
}

SessionDir::Options opts(const fs::path& d, const SessionFileIdentity& id, int stages = 0) {
    SessionDir::Options o;
    o.dir = d.string();
    o.id = id;
    o.write_delay = std::chrono::milliseconds(0);
    o.min_free_bytes = 0;
    if (std::getenv("SESSION_DIR_TEST_LOG")) o.log = [](const std::string& m) { std::fprintf(stderr, "  [dir] %s\n", m.c_str()); };
    for (int k = 0; k < stages; ++k) o.limits.stages.emplace_back();
    return o;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-sessiondir-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0xabcdef0123456789ull, 0x1122334455667788ull};

    const auto a = tokens(1000, 4000);
    auto a2v = a; for (int32_t t : tokens(9000, 500)) a2v.push_back(t);    // A, two turns later
    const auto b = tokens(50000, 3000);                                       // another conversation
    {
        SessionDir sd(opts(dir, id));
        sd.open();
        check(sd.files() == 0, "an empty folder");
        sd.write(std::make_shared<SavedConversation>(conv(a, 1)));
        sd.flush();
        check(sd.files() == 1 && session_files(dir) == 1, "a parked conversation is written");
        sd.write(std::make_shared<SavedConversation>(conv(a2v, 2)));
        sd.flush();
        check(sd.files() == 1 && session_files(dir) == 1, "the same conversation further on replaces its file");
        sd.write(std::make_shared<SavedConversation>(conv(b, 3)));
        sd.flush();
        check(sd.files() == 2, "another conversation gets its own file");

        auto prompt = a2v; prompt.push_back(77);
        int64_t t = 0;
        auto got = sd.fetch(prompt, {}, false, 0, t);
        check(got && got->live.ids == a2v && t == (int64_t) a2v.size(), "fetch: the conversation the prompt continues");
        check(!sd.fetch(prompt, {}, false, (int64_t) a2v.size(), t), "fetch: nothing when the cache already has as much");
        // the client rewrote the tail: the live tokens no longer match, the deepest checkpoint does
        auto rewritten = std::vector<int32_t>(a2v.begin(), a2v.begin() + 3000); rewritten.push_back(-5); rewritten.push_back(6);
        got = sd.fetch(rewritten, {}, false, 0, t);
        check(got && t == (int64_t) a2v.size() / 2, "fetch: from the deepest checkpoint after a rewritten tail");
        check(got->checkpoints.size() == 1, "files hold the deepest checkpoint only");
        check(!sd.fetch(tokens(70000, 100), {}, false, 0, t), "fetch: no match for an unknown conversation");
        check(!sd.fetch(prompt, {}, true, 0, t), "fetch: another steering mode does not match");

        // claim: a conversation taken back before its delayed write is never written
        SessionDir::Options slow = opts(dir, id);
        slow.write_delay = std::chrono::milliseconds(60000);
        SessionDir sd2(slow);
        sd2.open();
        auto c = std::make_shared<SavedConversation>(conv(tokens(80000, 2000), 4));
        sd2.write(c);
        sd2.claim(c.get());
        sd2.flush();
        check(session_files(dir) == 2, "claim drops a queued write");
    }
    // a parked conversation still inside its write delay at exit: written by flush, and by the destructor alone
    {
        SessionDir::Options slow = opts(dir, id);
        slow.write_delay = std::chrono::milliseconds(600000);
        const size_t before = session_files(dir);
        {
            SessionDir sd(slow);
            sd.open();
            sd.write(std::make_shared<SavedConversation>(conv(tokens(110000, 1500), 7)));
            check(session_files(dir) == before, "delay: not written yet");
            const auto t0 = std::chrono::steady_clock::now();
            sd.flush();
            check(session_files(dir) == before + 1 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30),
                  "flush writes a delayed image at once");
            sd.write(std::make_shared<SavedConversation>(conv(tokens(120000, 1500), 8)));
        }   // destructor only
        check(session_files(dir) == before + 2, "the destructor writes what is still queued");
        for (const auto& e : fs::directory_iterator(dir)) {   // leave the folder as the caps test expects it
            SessionPeek pk; std::string err;
            if (session_file_peek(e.path().string(), id, pk, err) && !pk.live.ids.empty() &&
                (pk.live.ids[0] == 110000 || pk.live.ids[0] == 120000)) fs::remove(e.path());
        }
    }
    // caps: the oldest file goes
    {
        SessionDir::Options o = opts(dir, id);
        o.max_files = 2;
        SessionDir sd(o);
        sd.open();
        sd.write(std::make_shared<SavedConversation>(conv(tokens(90000, 1000), 5)));
        sd.flush();
        check(sd.files() == 2 && session_files(dir) == 2, "max_files drops the oldest");
        int64_t t = 0;
        auto p = a2v; p.push_back(1);
        check(!sd.fetch(p, {}, false, 0, t), "the dropped one was the oldest (A)");
    }
    // a restart: the index comes back from the folder, the preload reads the newest first
    {
        SessionDir::Options o = opts(dir, id);
        o.preload_bytes = 1ull << 30;
        o.preload_files = 1;
        SessionDir sd(o);
        sd.open();
        check(sd.files() == 2, "reopen: both files indexed");
        sd.start_preload();
        std::vector<SavedConversation> pre;
        for (int i = 0; i < 200 && pre.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pre = sd.take_preloaded();
        }
        check(pre.size() == 1 && pre[0].live.ids == tokens(90000, 1000), "preload: the newest file, within its budget");
    }
    // a file of another identity is deleted at open
    {
        SessionDir sd(opts(dir, {id.model ^ 1, id.config}));
        sd.open();
        check(sd.files() == 0 && session_files(dir) == 0, "foreign files are deleted");
    }
    // a layer split: v2 files through the folder
    {
        SessionDir sd(opts(dir, id, 3));
        sd.open();
        sd.write(std::make_shared<SavedConversation>(conv(a, 6, 3)));
        sd.flush();
        int64_t t = 0;
        auto p = a; p.push_back(3);
        auto got = sd.fetch(p, {}, false, 0, t);
        check(got && got->stage_images.size() == 3 && got->stage_images[2].checkpoints.size() == 1,
              "split: every stage comes back with its deepest checkpoint");
    }
    // supersedes(): the rules on their own
    {
        SessionPeek older;
        older.live.ids = tokens(1, 100);
        older.deepest.ids = tokens(1, 60);
        auto live = head(tokens(1, 150), 1);
        check(SessionDir::supersedes(live, {}, older), "supersedes: a continuation");
        auto other = head(tokens(5000, 150), 1);
        check(!SessionDir::supersedes(other, {}, older), "supersedes: not another conversation");
        auto rewrite = head(tokens(1, 80), 1); rewrite.ids.push_back(-1);
        check(SessionDir::supersedes(rewrite, {head(tokens(1, 60), 2)}, older), "supersedes: the deepest checkpoint held");
        SessionPeek shared_root;
        shared_root.live.ids = tokens(1, 30);
        shared_root.live.ids.push_back(-9);
        shared_root.deepest.ids = tokens(1, 20);
        check(!SessionDir::supersedes(head(tokens(1, 150), 1), {head(tokens(1, 60), 2)}, shared_root),
              "supersedes: a branch whose deepest checkpoint the newer chain does not hold is kept");
    }
    fs::remove_all(dir);
    std::printf("session_dir_test: %d checks passed\n", checks);
    return 0;
}
