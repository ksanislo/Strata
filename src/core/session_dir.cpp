// --session-dir: a folder of session files, one per conversation (session_dir.hpp).
#include "strata/core/session_dir.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <random>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace strata::core {
namespace {

bool same_head(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs;
}

// a's tokens and images are the start of b's
bool head_prefix(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids.empty() || a.ids.size() > b.ids.size() || a.imgs.size() > b.imgs.size()) return false;
    return std::equal(a.ids.begin(), a.ids.end(), b.ids.begin()) &&
           std::equal(a.imgs.begin(), a.imgs.end(), b.imgs.begin());
}

// what a file written from `image` (deepest checkpoint only) holds, for the index - without reading it back
SessionPeek peek_of(const SavedConversation& image, uint64_t bytes) {
    SessionPeek p;
    p.live.ids = image.live.ids;
    p.live.imgs = image.live.imgs;
    p.cvec = image.cvec;
    p.bytes = bytes;
    p.version = image.stage_images.empty() ? 1 : 2;
    bool aligned = !image.checkpoints.empty();
    for (const auto& st : image.stage_images) aligned = aligned && st.checkpoints.size() == image.checkpoints.size();
    if (aligned) {
        const ConversationCheckpoint* deepest = session_deepest_checkpoint(image.checkpoints);   // the file's one
        p.deepest.ids = deepest->ids;
        p.deepest.imgs = deepest->imgs;
    }
    return p;
}

std::string random_name() {
    std::random_device rd;
    const uint64_t v = (uint64_t) rd() << 32 ^ rd();
    char b[40];
    std::snprintf(b, sizeof b, "%016" PRIx64 ".session", v);
    return b;
}

} // namespace

SessionDir::SessionDir(Options o) : o_(std::move(o)) {}

SessionDir::~SessionDir() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        flushing_ = true;   // the writer writes what is queued before it stops
    }
    cv_.notify_all();
    if (preloader_.joinable()) preloader_.join();
    if (writer_.joinable()) writer_.join();
}

void SessionDir::say(const std::string& m) const {
    if (o_.log) o_.log(m);
}

bool SessionDir::supersedes(const ConversationCheckpoint& newer_live, const std::vector<ConversationCheckpoint>& newer_chain,
                            const SessionPeek& older) {
    // the same conversation further on: its live tokens continue the older file's
    if (head_prefix(older.live, newer_live)) return true;
    // or the older file's deepest checkpoint is held by the newer chain (#342's rule: a client that rewrote the tail)
    if (older.deepest.ids.empty()) return false;
    if (same_head(older.deepest, newer_live)) return true;
    for (const auto& c : newer_chain)
        if (same_head(older.deepest, c)) return true;
    return false;
}

void SessionDir::open() {
    std::error_code ec;
    fs::create_directories(o_.dir, ec);
    std::vector<std::pair<fs::file_time_type, Entry>> found;
    for (const auto& de : fs::directory_iterator(o_.dir, ec)) {
        if (!de.is_regular_file() || de.path().extension() != ".session") continue;
        Entry e;
        e.path = de.path().string();
        std::string err;
        if (!session_file_peek(e.path, o_.id, e.peek, err)) {
            say("drop " + e.path + " (" + err + ")");
            fs::remove(de.path(), ec);
            continue;
        }
        found.emplace_back(de.last_write_time(ec), std::move(e));
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& f : found) {
        f.second.seq = ++seq_;
        entries_.push_back(std::move(f.second));
    }
    enforce_caps_locked();
    uint64_t total = 0;
    for (const auto& e : entries_) total += e.peek.bytes;
    say(std::to_string(entries_.size()) + " conversation file(s), " + std::to_string(total >> 20) + " MiB in " + o_.dir);
    writer_ = std::thread([this] { writer_loop(); });
}

void SessionDir::start_preload() {
    if (o_.preload_bytes == 0 || o_.preload_files == 0) return;
    preloader_ = std::thread([this] { preload_loop(); });
}

void SessionDir::write(std::shared_ptr<const SavedConversation> image) {
    if (!image || image->live.ids.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        // a queued image of the same conversation, older: this one replaces it (never written twice)
        jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const Job& j) {
            return SessionDir::supersedes(image->live, image->checkpoints, peek_of(*j.image, 0));
        }), jobs_.end());
        jobs_.push_back({std::move(image), std::chrono::steady_clock::now() + o_.write_delay});
    }
    cv_.notify_all();
}

void SessionDir::claim(const SavedConversation* image) {
    std::unique_lock<std::mutex> lk(mu_);
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const Job& j) { return j.image.get() == image; }),
                jobs_.end());
    cv_.wait(lk, [&] { return writing_ != image; });
}

void SessionDir::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    flushing_ = true;
    cv_.notify_all();
    cv_.wait(lk, [&] { return jobs_.empty() && writing_ == nullptr; });
    flushing_ = stop_;
}

size_t SessionDir::files() const {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_.size();
}

uint64_t SessionDir::bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t n = 0;
    for (const auto& e : entries_) n += e.peek.bytes;
    return n;
}

void SessionDir::writer_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        if (jobs_.empty()) {
            if (stop_) return;
            cv_.wait(lk);
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!flushing_ && jobs_.front().due > now) {
            cv_.wait_until(lk, jobs_.front().due);
            continue;
        }
        Job job = std::move(jobs_.front());
        jobs_.pop_front();
        writing_ = job.image.get();
        lk.unlock();

        const std::string path = (fs::path(o_.dir) / random_name()).string();
        SessionWriteOptions wo;
        wo.deepest_checkpoint_only = true;
        wo.min_free_bytes = o_.min_free_bytes;
        size_t written = 0;
        std::string err;
        SessionStatus st;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = session_file_write(path, *job.image, o_.id, written, err, wo, &st);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        lk.lock();
        writing_ = nullptr;
        if (ok) {
            add_written_locked(path, *job.image, written, s, "wrote");
        } else {
            say("write failed: " + err);
        }
        job.image.reset();   // an image evicted from the cache meanwhile is freed here, outside the cache's budget
        cv_.notify_all();
    }
}

void SessionDir::add_written_locked(const std::string& path, const SavedConversation& image, size_t written, double s,
                                    const char* what, const std::vector<ConversationCheckpoint>* chain) {
    Entry e;
    e.path = path;
    e.peek = peek_of(image, written);
    e.seq = ++seq_;
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        if (entries_[i].state != State::loading &&
            supersedes(image.live, chain ? *chain : image.checkpoints, entries_[i].peek)) {
            drop_entry_locked(i, "superseded");
            ++dropped;
        } else {
            ++i;
        }
    }
    char b[200];
    std::snprintf(b, sizeof b, "%s %zu tokens, %zu MiB in %.1f s (%zu older file(s) of it dropped)", what,
                  image.live.ids.size(), written >> 20, s, dropped);
    say(b);
    entries_.push_back(std::move(e));
    enforce_caps_locked();
}

bool SessionDir::write_now(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                           const std::vector<std::vector<SessionKvSource>>& stage_kv, size_t& bytes, std::string& error,
                           const std::vector<ConversationCheckpoint>* chain) {
    const std::string path = (fs::path(o_.dir) / random_name()).string();
    SessionWriteOptions wo;
    wo.min_free_bytes = o_.min_free_bytes;
    SessionStatus st;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = meta.stage_images.empty() ? session_file_write(path, meta, kv, o_.id, bytes, error, wo, &st)
                                              : session_file_write(path, meta, kv, stage_kv, o_.id, bytes, error, wo, &st);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) { say("write failed: " + error); return false; }
    std::lock_guard<std::mutex> lk(mu_);
    add_written_locked(path, meta, bytes, s, "wrote (directly, low RAM)", chain);
    return true;
}

void SessionDir::preload_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return stop_ || !fetching_; });   // a request's own read goes first
        if (stop_) return;
        Entry* pick = nullptr;
        for (auto& e : entries_)
            if (e.state == State::idle && preloaded_files_ < o_.preload_files &&
                preloaded_bytes_ + e.peek.bytes <= o_.preload_bytes && (!pick || e.seq > pick->seq))
                pick = &e;
        if (!pick) return;
        pick->state = State::loading;
        const std::string path = pick->path;
        lk.unlock();
        SavedConversation image;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = read_file(path, image, err);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lk.lock();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].path != path) continue;
            if (ok) {
                entries_[i].state = State::preloaded;
                preloaded_bytes_ += entries_[i].peek.bytes;
                ++preloaded_files_;
                char b[200];
                std::snprintf(b, sizeof b, "preloaded %zu tokens in %.1f s", image.live.ids.size(), s);
                say(b);
                ready_.push_back({path, std::move(image)});
            } else {
                say("preload failed: " + err);
                drop_entry_locked(i, "unreadable");
            }
            break;
        }
        cv_.notify_all();
    }
}

std::vector<SavedConversation> SessionDir::take_preloaded() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<SavedConversation> out;
    for (auto& r : ready_) {
        for (auto& e : entries_)
            if (e.path == r.path) e.state = State::idle;   // the caller's cache holds it now; the file stays
        out.push_back(std::move(r.image));
    }
    ready_.clear();
    return out;
}

int64_t SessionDir::match(const Entry& e, const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                          bool cvec) const {
    if (e.peek.cvec != cvec) return 0;
    const int64_t t = conversation_prefix(e.peek.live, ids, imgs);
    return t > 0 ? t : conversation_prefix(e.peek.deepest, ids, imgs);
}

std::optional<SavedConversation> SessionDir::fetch(const std::vector<int32_t>& ids,
                                                   const std::vector<ConversationImageKey>& imgs, bool cvec,
                                                   int64_t beat, int64_t& tokens) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        size_t best = entries_.size();
        int64_t best_t = beat;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const int64_t t = match(entries_[i], ids, imgs, cvec);
            if (t > best_t) { best = i; best_t = t; }
        }
        if (best == entries_.size()) return std::nullopt;
        Entry& e = entries_[best];
        const std::string path = e.path;
        if (e.state == State::loading) {   // the preload is reading it: wait for that read
            cv_.wait(lk, [&] {
                for (const auto& x : entries_) if (x.path == path) return x.state != State::loading;
                return true;
            });
            continue;
        }
        if (e.state == State::preloaded) {
            for (size_t r = 0; r < ready_.size(); ++r) {
                if (ready_[r].path != path) continue;
                SavedConversation out = std::move(ready_[r].image);
                ready_.erase(ready_.begin() + (std::ptrdiff_t) r);
                e.state = State::idle;
                e.seq = ++seq_;
                tokens = best_t;
                return out;
            }
            e.state = State::idle;
            continue;
        }
        e.state = State::loading;
        fetching_ = true;
        lk.unlock();
        SavedConversation image;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = read_file(path, image, err);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lk.lock();
        fetching_ = false;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].path != path) continue;
            if (ok) {
                entries_[i].state = State::idle;
                entries_[i].seq = ++seq_;
            } else {
                say("read failed: " + err);
                drop_entry_locked(i, "unreadable");
            }
            break;
        }
        cv_.notify_all();
        if (ok) {
            char b[160];
            std::snprintf(b, sizeof b, "read %zu tokens for this request in %.1f s", image.live.ids.size(), s);
            say(b);
            tokens = best_t;
            return image;
        }
    }
}

bool SessionDir::read_file(const std::string& path, SavedConversation& image, std::string& error) const {
    size_t bytes = 0;
    SessionStatus st;
    return session_file_map(path, o_.id, image, bytes, error, o_.limits, &st);   // K/V as views: no RAM copy
}

void SessionDir::drop_entry_locked(size_t i, const char* why) {
    std::error_code ec;
    fs::remove(entries_[i].path, ec);
    say(std::string("deleted ") + fs::path(entries_[i].path).filename().string() + " (" + why + ")");
    entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
}

void SessionDir::enforce_caps_locked() {
    for (;;) {
        uint64_t total = 0;
        for (const auto& e : entries_) total += e.peek.bytes;
        if (entries_.size() <= o_.max_files && total <= o_.max_bytes) return;
        size_t oldest = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].state == State::idle && (oldest == entries_.size() || entries_[i].seq < entries_[oldest].seq))
                oldest = i;
        if (oldest == entries_.size()) return;   // everything left is being read: next time
        drop_entry_locked(oldest, "over the folder's cap");
    }
}

} // namespace strata::core
