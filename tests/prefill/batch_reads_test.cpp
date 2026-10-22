// STRATA_BATCH_READS: how the prompt path's queue worker picks the prompts of one shared chunk and how many tokens each
// gets (Prefill::plan_batch / batch_shares).  CPU only: no device is touched.
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <deque>
#include <memory>
#include <vector>

using strata::prefill::Prefill;
using strata::prefill::PromptJob;

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
std::vector<int64_t> v(std::initializer_list<int64_t> x) { return std::vector<int64_t>(x); }
std::shared_ptr<PromptJob> job(strata::core::SessionState* s, int64_t n, int64_t taken = 0) {
    auto j = std::make_shared<PromptJob>();
    j->tokens.assign((size_t) n, 7);
    j->sessions = {s, s};
    j->taken = taken;
    return j;
}
}  // namespace

int main() {
    // shares: even, what one leaves goes to the others, never past the budget, a remainder below one each is left
    expect(Prefill::batch_shares(v({2048, 2048, 2048}), 2048) == v({682, 682, 682}), "three equal prompts: equal shares");
    expect(Prefill::batch_shares(v({100, 2048, 2048}), 2048) == v({100, 974, 974}), "a short prompt's rest goes round");
    expect(Prefill::batch_shares(v({100, 200}), 2048) == v({100, 200}), "both fit whole");
    expect(Prefill::batch_shares(v({5000, 300}), 2048) == v({1748, 300}), "a long one takes what the short leaves");
    expect(Prefill::batch_shares(v({1, 1, 1}), 2) == v({0, 0, 0}), "a budget below the prompts' count: none");
    int64_t total = 0;
    for (int64_t g : Prefill::batch_shares(v({999, 1001, 37, 4096}), 2048)) total += g;
    expect(total <= 2048 && total >= 2048 - 3, "the budget is used up to a remainder below one each");

    strata::core::SessionState sa, sb, sc, sd, se;
    {   // the first queued piece of each session, in queue order; a session whose first piece cannot go is passed over
        std::deque<std::shared_ptr<PromptJob>> q;
        q.push_back(job(&sa, 2048));
        q.push_back(job(&sa, 2048));   // a's next piece waits for its first
        q.push_back(job(&sb, 300));
        auto c = job(&sc, 500);
        static const float row[1] = {0.0f};
        static const float* const rows[1] = {row};
        c->embd_rows = rows;   // a picture: read alone
        q.push_back(c);
        q.push_back(job(&sc, 400));    // behind c's picture piece
        q.push_back(job(&sd, 100));
        q.push_back(job(&se, 100));
        const Prefill::BatchPlan p = Prefill::plan_batch(q, 4, 2048);
        expect(p.at == std::vector<size_t>({0, 2, 5, 6}), "a, b, d, e (not a's second piece, nothing of c)");
        expect(p.got == v({1548, 300, 100, 100}), "the shares of a, b, d, e");
        const Prefill::BatchPlan p2 = Prefill::plan_batch(q, 2, 2048);
        expect(p2.at == std::vector<size_t>({0, 2}) && p2.got == v({1748, 300}), "at most two");
    }
    {   // a sliced prompt goes on from what it gave; a lone session or a front that cannot share: no batch
        std::deque<std::shared_ptr<PromptJob>> q;
        q.push_back(job(&sa, 2048, 2000));
        q.push_back(job(&sb, 2048));
        const Prefill::BatchPlan p = Prefill::plan_batch(q, 4, 2048);
        expect(p.got == v({48, 2000}), "the rest of a sliced prompt");
        std::deque<std::shared_ptr<PromptJob>> lone;
        lone.push_back(job(&sa, 2048));
        lone.push_back(job(&sa, 2048));
        expect(Prefill::plan_batch(lone, 4, 2048).at.empty(), "one session: read the usual way (full chunks)");
        std::deque<std::shared_ptr<PromptJob>> main_first;
        main_first.push_back(job(nullptr, 100));
        main_first.push_back(job(&sb, 100));
        main_first.push_back(job(&sc, 100));
        expect(Prefill::plan_batch(main_first, 4, 2048).at.empty(), "a front prompt without sessions of its own goes alone");
        std::deque<std::shared_ptr<PromptJob>> stopped;
        stopped.push_back(job(&sa, 100));
        stopped.push_back(job(&sb, 100));
        stopped.push_back(job(&sc, 100));
        stopped[1]->should_stop = [] { return true; };
        const Prefill::BatchPlan p3 = Prefill::plan_batch(stopped, 4, 2048);
        expect(p3.at == std::vector<size_t>({0, 2}), "a cancelled prompt is left to end the usual way");
        expect(Prefill::plan_batch(q, 1, 2048).at.empty(), "off below two");
    }
    if (fails == 0) std::printf("batch_reads_test: ok\n");
    return fails == 0 ? 0 : 1;
}
