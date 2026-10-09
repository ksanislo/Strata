#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
// A VIEW (session files read through a mapping, low-RAM restores) reads bytes it does not own: it counts no RAM
// (bytes() == 0), a read visit hands out the mapped bytes, a writing visit fails, and resizing copies it first.
class ConversationBuffer {
public:
    static ConversationBuffer view(const uint8_t* data, size_t n, std::shared_ptr<const void> keep) {
        ConversationBuffer b;
        b.ext_ = data;
        b.size_ = n;
        b.keep_ = std::move(keep);
        return b;
    }
    bool external() const { return ext_ != nullptr; }
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)),
          ext_(std::exchange(other.ext_, nullptr)), keep_(std::move(other.keep_)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        ext_ = std::exchange(other.ext_, nullptr);
        keep_ = std::move(other.keep_);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        if (ext_) return 0;   // the mapping's pages are the OS file cache's, not this process's RAM
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        if (ext_) return std::max(n, size_);   // a resize copies the view into owned segments first
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (ext_) own();
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        if (ext_ || other.ext_) {   // a view on either side: compare through reads
            std::vector<uint8_t> a(std::min(size_, segment_bytes)), b(a.size());
            for (size_t at = 0; at < size_; at += a.size()) {
                const size_t n = std::min(a.size(), size_ - at);
                if (!read(a.data(), at, n) || !other.read(b.data(), at, n) || std::memcmp(a.data(), b.data(), n))
                    return false;
            }
            return true;
        }
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    // a view becomes owned segments (a resize of a view)
    void own() {
        const uint8_t* src = ext_;
        const size_t n = size_;
        ext_ = nullptr;
        size_ = 0;
        segments_.clear();
        resize(n);
        visit(0, n, [&](uint8_t* p, size_t c, size_t at) { std::memcpy(p, src + at, c); return true; });
        keep_.reset();
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        if (self.ext_) {
            if constexpr (!std::is_const_v<Self>) {
                return false;   // a view is read-only: writing into it fails (the caller copies or captures fresh)
            } else {
                while (count) {   // in segment-sized steps, as the owned storage hands them out
                    const size_t n = std::min(count, segment_bytes);
                    if (!fn(self.ext_ + offset, n, offset)) return false;
                    offset += n; count -= n;
                }
                return true;
            }
        }
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
    const uint8_t* ext_ = nullptr;      // a view's bytes (none: owned segments)
    std::shared_ptr<const void> keep_;  // keeps a view's mapping alive
};

} // namespace strata::core
