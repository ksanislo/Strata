// tests/program/stage_cores_test.cpp - STRATA_STAGE_SERVE's core plan (header-only, CPU).
//
//   1. 4 GPUs 2+2 over two 10-core nodes, the host on core 0: node 0's 9 cores go 4/5 to stages 0/1 by their layers;
//   2. --pool-workers caps the workers of all stages together, by weight;
//   3. a node that is not known puts every stage on all the cores;
//   4. fewer cores than stages: they share, and every stage still has a worker;
//   5. the cpulist parser.
#include "strata/program/stage_cores.hpp"

#include <cstdio>

namespace sp = strata::program;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::vector<int> range(int a, int b) {
    std::vector<int> v;
    for (int i = a; i < b; ++i) v.push_back(i);
    return v;
}
}  // namespace

int main() {
    std::printf("stage_cores_test\n");
    std::map<int, std::vector<int>> box{{-1, range(0, 20)}, {0, range(0, 10)}, {1, range(10, 20)}};
    const std::vector<int> nodes{0, 0, 1, 1}, layers{11, 13, 12, 12};
    {
        const auto p = sp::plan_stage_cores(nodes, layers, box, {0, 20}, 0);
        check(p.size() == 4 && p[0].node == 0 && p[2].node == 1, "each stage on its GPU's node");
        check(p[0].service == 1 && p[0].workers == std::vector<int>({2, 3, 4}), "stage 0: service 1, workers 2-4");
        check(p[1].service == 5 && p[1].workers == std::vector<int>({6, 7, 8, 9}), "stage 1 (more layers): service 5, workers 6-9");
        check(p[2].service == 10 && p[2].workers.size() == 4 && p[3].service == 15 && p[3].workers.size() == 4,
              "node 1: two shares of 5 cores");
    }
    {
        const auto p = sp::plan_stage_cores(nodes, layers, box, {0, 20}, 10);
        size_t t = 0;
        for (const auto& s : p) t += s.workers.size();
        check(t == 10 && p[0].workers.size() == 2 && p[1].workers.size() == 3, "--pool-workers 10: 10 workers by weight");
        const auto q = sp::plan_stage_cores(nodes, layers, box, {0, 20}, 19);
        t = 0;
        for (const auto& s : q) t += s.workers.size();
        check(t == 15, "--pool-workers 19: capped at the 15 cores left");
    }
    {
        const auto p = sp::plan_stage_cores({0, 0, -1, 1}, layers, box, {0}, 0);
        check(p[0].node == -1 && p[3].node == -1 && p[0].service == 1, "an unknown node: every stage on all cores");
        size_t t = 0;
        for (const auto& s : p) t += s.workers.size() + 1;
        check(t == 19, "all 19 cores used");
    }
    {
        std::map<int, std::vector<int>> small{{-1, {0, 1}}, {0, {0, 1}}};
        const auto p = sp::plan_stage_cores({0, 0, 0}, {1, 1, 1}, small, {}, 0);
        check(p[0].service == 0 && p[1].service == 1 && p[2].service == 0, "fewer cores than stages: shared");
        check(p[2].workers == std::vector<int>({0}), "a one-core share: the worker shares the service core");
    }
    {   // service threads on SMT siblings: every physical core works
        std::map<int, int> sib;
        for (int c = 0; c < 20; ++c) sib[c] = c + 20;
        const auto p = sp::plan_stage_cores(nodes, layers, box, {0, 20}, 0, &sib);
        check(p[0].service == 21 && p[0].workers == std::vector<int>({1, 2, 3, 4}), "sibling: stage 0 service 21, workers 1-4");
        check(p[1].service == 25 && p[1].workers.size() == 5 && p[3].service == 35 && p[3].workers.size() == 5,
              "sibling: 19 physical cores all work");
    }
    {
        const auto w = sp::split_by_weight(7, {11, 13}, {100, 100});
        check(w[0] == 3 && w[1] == 4, "split_by_weight: 7 by 11:13 is 3+4");
        const auto c = sp::split_by_weight(6, {1, 1, 1}, {1, 5, 5});
        check(c[0] == 1 && c[1] == 3 && c[2] == 2, "split_by_weight: a capped share's rest goes to the others");
    }
    {
        std::vector<int> v;
        check(sp::parse_cpulist("0-2,20-21\n", v) && v == std::vector<int>({0, 1, 2, 20, 21}), "cpulist 0-2,20-21");
        check(sp::parse_cpulist("7", v) && v == std::vector<int>({7}), "cpulist 7");
        check(!sp::parse_cpulist("3-1", v) && !sp::parse_cpulist("", v) && !sp::parse_cpulist("a", v), "cpulist refusals");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
