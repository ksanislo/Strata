// A folder of session files, one per conversation (--session-dir): the conversations the engine parks are written
// to it behind the engine's back (a writer thread, after a short delay so a conversation taken back at once is never
// written), and its files come back into the conversation cache - in the background after the start, newest first,
// or at once when a request continues one.  Every file is a session file (conversation_file.hpp) bound to this
// runtime's identity; a file this runtime refuses is deleted.  Pure host code: no CUDA.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class SessionDir {
public:
    struct Options {
        std::string dir;
        SessionFileIdentity id;
        SessionReadLimits limits;              // this runtime's bounds (its stages included); admit = RAM preflight
        size_t max_files = 8;                  // the oldest files beyond these caps are deleted
        uint64_t max_bytes = 64ull << 30;
        uint64_t preload_bytes = 0;            // the background preload stops before this many bytes (0: no preload)
        size_t preload_files = 0;
        std::chrono::milliseconds write_delay{30000};   // a parked image is written this long after it was parked
        uint64_t min_free_bytes = 4ull << 30;  // a write is refused when the disk would keep less than this free
        std::function<void(const std::string&)> log;
    };
    explicit SessionDir(Options o);
    ~SessionDir();                             // stops the preload, writes every queued image, then stops
    SessionDir(const SessionDir&) = delete;
    SessionDir& operator=(const SessionDir&) = delete;

    // Index the folder (peek): the files this runtime reads; any other *.session file is deleted.  Starts the writer.
    void open();
    // The background preload: newest file first, until preload_bytes / preload_files.
    void start_preload();
    // A parked image, written after write_delay (or at flush).  Files it supersedes (the same conversation, older)
    // are deleted once it is on disk.
    void write(std::shared_ptr<const SavedConversation> image);
    // Low RAM: write a conversation NOW from streamed sources (no host copy of its K/V: conversation_snapshot_sources;
    // one list per stage image in stage_kv), on the caller's thread; indexed and superseding like a parked image.
    // `chain`: the conversation's whole checkpoint chain (token ids and images are used), for deciding which older
    // files it supersedes - the file itself holds only the deepest checkpoint (meta.checkpoints).
    bool write_now(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                   const std::vector<std::vector<SessionKvSource>>& stage_kv, size_t& bytes, std::string& error,
                   const std::vector<ConversationCheckpoint>* chain = nullptr);
    // Before the caller moves `image` out (take): a queued write of it is dropped, a running one is waited for.
    void claim(const SavedConversation* image);
    // The images the preload has read, for the caller's cache (each file's image once).
    std::vector<SavedConversation> take_preloaded();
    // The on-disk conversation that continues `ids` furthest when it beats `beat` tokens, read now (or waited for
    // while the preload reads it).  nullopt when none beats it, or the read failed (logged; the file is deleted).
    std::optional<SavedConversation> fetch(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                                           bool cvec, int64_t beat, int64_t& tokens);
    // Write every queued image now and wait until they are on disk.
    void flush();
    size_t files() const;
    uint64_t bytes() const;

    // Exposed for tests: does `newer` supersede `older` (the same conversation, further on)?
    static bool supersedes(const ConversationCheckpoint& newer_live, const std::vector<ConversationCheckpoint>& newer_chain,
                           const SessionPeek& older);

private:
    enum class State { idle, loading, preloaded };
    struct Entry {
        std::string path;
        SessionPeek peek;
        uint64_t seq = 0;                      // recency: higher is newer
        State state = State::idle;
    };
    struct Job {
        std::shared_ptr<const SavedConversation> image;
        std::chrono::steady_clock::time_point due;
    };
    struct Ready {
        std::string path;
        SavedConversation image;
    };

    void say(const std::string& m) const;
    void add_written_locked(const std::string& path, const SavedConversation& image, size_t written, double s,
                            const char* what, const std::vector<ConversationCheckpoint>* chain = nullptr);
    void writer_loop();
    void preload_loop();
    bool read_file(const std::string& path, SavedConversation& image, std::string& error) const;
    void drop_entry_locked(size_t i, const char* why);
    void enforce_caps_locked();
    int64_t match(const Entry& e, const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                  bool cvec) const;

    Options o_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Entry> entries_;
    std::deque<Job> jobs_;
    std::vector<Ready> ready_;
    const SavedConversation* writing_ = nullptr;
    uint64_t seq_ = 0, preloaded_bytes_ = 0;
    size_t preloaded_files_ = 0;
    bool stop_ = false, flushing_ = false, fetching_ = false;
    std::thread writer_, preloader_;
};

} // namespace strata::core
