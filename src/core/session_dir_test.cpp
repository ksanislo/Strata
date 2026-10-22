// CPU-only tests for --session-dir (include/strata/core/session_dir.hpp): write-behind, supersede, claim, caps,
// fetch, preload, foreign files.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/session_dir.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
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
        check(sd.files() == 2 && session_files(dir) == 2, "max_files drops the least recently used");
        int64_t t = 0;
        auto pb = b; pb.push_back(1);
        check(!sd.fetch(pb, {}, false, 0, t), "the dropped one was B: written after A, but A was RESUMED since");
        auto p = a2v; p.push_back(1);
        check(sd.fetch(p, {}, false, 0, t).has_value(), "A, resumed most recently before the new one, is kept");
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
        check(pre.size() == 1 && pre[0].live.ids == a2v, "preload: the most recently used file (A, resumed last), within its budget");
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
    // low RAM: a conversation written NOW from streamed sources (no image copy), then fetched as a mapped view
    {
        SessionDir sd(opts(dir, id));
        sd.open();
        const SavedConversation full = conv(tokens(130000, 2500), 9);
        SavedConversation meta = full;
        meta.kv.clear();
        meta.checkpoints = {full.checkpoints[1]};   // the engine passes the deepest only
        std::vector<SessionKvSource> src;
        for (const auto& k : full.kv) {
            SessionKvSource x;
            x.format = k.format; x.cells = k.cells; x.heads = k.heads; x.head_dim = k.head_dim;
            x.page_size = k.page_size; x.pooled_rows = k.pooled_rows; x.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) x.sizes[i] = parts[i]->size();
            x.read = [parts](size_t part, size_t at, void* dst, size_t n) { return parts[part]->read(dst, at, n); };
            src.push_back(std::move(x));
        }
        size_t b = 0;
        std::string err;
        const size_t before = sd.files();
        check(sd.write_now(meta, src, {}, b, err) && sd.files() == before + 1, "write_now: written and indexed");
        int64_t t = 0;
        auto p = full.live.ids; p.push_back(1);
        auto got = sd.fetch(p, {}, false, 0, t);
        check(got && got->live.ids == full.live.ids && got->kv.size() == 1 && got->kv[0].k.external() &&
              got->kv[0].k == full.kv[0].k, "write_now: fetched back as a mapped view with the same K/V");
    }
    // age: a file not written or resumed for max_age is deleted at open
    {
        SessionDir::Options o = opts(dir, id);
        SessionDir sd(o);
        sd.open();
        sd.write(std::make_shared<SavedConversation>(conv(tokens(170000, 2000), 14)));
        sd.flush();
        fs::path newest;
        for (const auto& e : fs::directory_iterator(dir))
            if (newest.empty() || fs::last_write_time(e.path()) > fs::last_write_time(newest)) newest = e.path();
        fs::last_write_time(newest, fs::file_time_type::clock::now() - std::chrono::hours(24 * 8));
        const size_t n = session_files(dir);
        SessionDir::Options aged = opts(dir, id);
        aged.max_age = std::chrono::hours(24 * 7);
        SessionDir sd2(aged);
        sd2.open();
        check(session_files(dir) == n - 1 && !fs::exists(newest), "max_age: an 8-day-old file is deleted, the rest kept");
    }
    // the server's notes: an event per write / read / deletion (with its reason) and the folder's state as JSON
    {
        std::vector<std::string> notes;
        std::mutex nm;
        SessionDir::Options o = opts(dir, id);
        o.event = [&](const std::string& m) { std::lock_guard<std::mutex> lk(nm); notes.push_back(m); };
        SessionDir sd(o);
        sd.open();
        const auto q = tokens(180000, 3000);
        sd.write(std::make_shared<SavedConversation>(conv(q, 15)));
        sd.flush();
        auto q2 = q; for (int32_t t : tokens(181000, 200)) q2.push_back(t);
        sd.write(std::make_shared<SavedConversation>(conv(q2, 16)));
        sd.flush();
        auto pq = q2; pq.push_back(1);
        int64_t t = 0;
        check(sd.fetch(pq, {}, false, 0, t).has_value(), "notes: fetch for the read event");
        std::lock_guard<std::mutex> lk(nm);
        auto has = [&](const char* a, const char* b = nullptr) {
            for (const auto& n : notes) if (n.find(a) != std::string::npos && (!b || n.find(b) != std::string::npos)) return true;
            return false;
        };
        check(has("sess_event=opened"), "notes: opened");
        check(has("sess_event=wrote sess_tokens=3000", "sess_why=background"), "notes: a background write");
        check(has("sess_event=deleted", "sess_why=superseded"), "notes: the older copy replaced");
        check(has("sess_event=read sess_tokens=3200", "sess_why=request"), "notes: a read for a request");
        const std::string& last = notes.back();
        check(last.rfind("sess_json={\"files\":", 0) == 0 && last.find(' ') == std::string::npos &&
              last.find("\"conversations\":[{\"tokens\":3200") != std::string::npos,
              "notes: the state JSON, most recent first, no spaces");
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
