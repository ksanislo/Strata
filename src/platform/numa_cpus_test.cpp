// numa_cpus.hpp: cpulist parsing (synthetic) and this machine's node CPUs, if it lists any.
#include "strata/platform/numa_cpus.hpp"

#include <cstdio>

using strata::platform::parse_cpulist;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(parse_cpulist("0-9,20-29\n").size() == 20);
    CHECK(parse_cpulist("3") == std::vector<int>{3});
    CHECK((parse_cpulist("1,4-5") == std::vector<int>{1, 4, 5}));
    CHECK(parse_cpulist("").empty());
    CHECK(parse_cpulist("x,5-2,7").size() == 1);
    const auto n0 = strata::platform::numa_node_cpus(0);
    std::printf("node 0: %zu CPUs, node 1: %zu CPUs\n", n0.size(), strata::platform::numa_node_cpus(1).size());
    CHECK(strata::platform::numa_node_cpus(-1).empty());
    if (!n0.empty()) CHECK(strata::platform::pin_this_thread(n0));
    if (fails == 0) std::printf("numa_cpus_test: ok\n");
    return fails == 0 ? 0 : 1;
}
