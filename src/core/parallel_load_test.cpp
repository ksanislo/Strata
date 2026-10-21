// STRATA_PARALLEL_LOAD's GGUF reader against the reference one (CPU only, synthetic shards): every byte of the arena
// the same, with one unpinned group and with the layers split over two NUMA groups, a role in a second shard included.
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <random>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";
    const std::string a = dir + "/parallel_load_test-1.gguf", b = dir + "/parallel_load_test-2.gguf";
    strata::kernels::cpu::ExpertLayout lay;
    lay.native = true;
    // STRATA_PARALLEL_LOAD_BENCH=1: 48 layers x 64 experts of ~2 MiB (6 GiB) instead, timed (fresh anonymous memory)
    const bool bench = std::getenv("STRATA_PARALLEL_LOAD_BENCH") != nullptr;
    lay.n_layers = bench ? 48 : 6;
    lay.n_expert = bench ? 64 : 8;
    std::mt19937_64 rng(7);
    std::vector<uint8_t> fa(1 << 16), fb;   // a header-ish lead-in, then the tensors
    for (auto& x : fa) x = (uint8_t) rng();
    uint64_t total = 0;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        strata::kernels::cpu::NativeFmt fm;
        fm.up_off = (bench ? 700000 : 4096) + 512 * (size_t) l;   // gate and up rows: up_off bytes each
        fm.down_off = 2 * fm.up_off;
        fm.bytes = fm.down_off + 3000 + 64 * (size_t) l;
        lay.fmt.push_back(fm);
        lay.bytes.push_back(fm.bytes);
        lay.offset.push_back(total);
        total += fm.bytes * (uint64_t) lay.n_expert;
        const uint64_t per[3] = {fm.up_off, fm.up_off, fm.bytes - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const bool second = l == 4 && r == 2;   // a role in another shard
            std::vector<uint8_t>& f = second ? fb : fa;
            lay.gguf_off.push_back(f.size());
            lay.gguf_file.push_back(second ? "parallel_load_test-2.gguf" : "");
            for (uint64_t i = 0; i < per[r] * (uint64_t) lay.n_expert; ++i) f.push_back((uint8_t) rng());
        }
    }
    lay.total = total;
    lay.max_blob = 0;
    for (auto x : lay.bytes) lay.max_blob = std::max<uint64_t>(lay.max_blob, x);
    for (auto& [path, data] : {std::pair<std::string, std::vector<uint8_t>*>{a, &fa}, {b, &fb}}) {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr && std::fwrite(data->data(), 1, data->size(), f) == data->size());
        if (f) std::fclose(f);
    }
    if (bench) {
        auto fresh = [&] { return (uint8_t*) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); };
        auto secs = [](auto t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        uint8_t* m = fresh();
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t o = 0; o < total; o += 4096) m[o] = 0;   // what the registration's fault-in does, one thread
        const double fault = secs(t0);
        t0 = std::chrono::steady_clock::now();
        strata::core::load_experts_gguf(a, m, lay, 6);
        const double fill6 = secs(t0);
        munmap(m, total);
        const char* tv = std::getenv("STRATA_PARALLEL_LOAD_THREADS");
        const int per = tv ? std::atoi(tv) : 8;
        for (const std::vector<int>& nodes : {std::vector<int>{}, std::vector<int>(48, 0)}) {
            std::vector<int> nn = nodes;
            if (!nn.empty()) for (int l = 24; l < 48; ++l) nn[(size_t) l] = 1;
            m = fresh();
            t0 = std::chrono::steady_clock::now();
            const auto sp = strata::core::load_experts_gguf_parallel(a, m, lay, nn, per);
            std::printf("parallel (%s, %d per group): %.2f s; today: fault %.2f s + 6 readers %.2f s (%.2f GiB)\n",
                        nn.empty() ? "unpinned" : "2 nodes", per, secs(t0), fault, fill6, total / 1073741824.0);
            CHECK(sp.ok);
            munmap(m, total);
        }
        std::remove(a.c_str());
        std::remove(b.c_str());
        return fails == 0 ? 0 : 1;
    }
    std::vector<uint8_t> ref(total, 0), got(total, 0);
    const auto st = strata::core::load_experts_gguf(a, ref.data(), lay, 2);
    CHECK(st.ok && st.bytes == total);
    for (const std::vector<int>& nodes : {std::vector<int>{}, std::vector<int>{0, 0, 0, 1, 1, 1}}) {
        std::fill(got.begin(), got.end(), 0);
        const auto sp = strata::core::load_experts_gguf_parallel(a, got.data(), lay, nodes, 3);
        CHECK(sp.ok && sp.bytes == total);
        CHECK(std::memcmp(ref.data(), got.data(), total) == 0);
    }
    const auto bad = strata::core::load_experts_gguf_parallel(dir + "/no-such.gguf", got.data(), lay, {}, 2);
    CHECK(!bad.ok && !bad.error.empty());
    std::remove(a.c_str());
    std::remove(b.c_str());
    if (fails == 0) std::printf("parallel_load_test: ok (%llu bytes)\n", (unsigned long long) total);
    return fails == 0 ? 0 : 1;
}
