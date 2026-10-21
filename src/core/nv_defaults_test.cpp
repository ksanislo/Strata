// src/core/nv_defaults_test.cpp - the per-architecture exact-switch tables (nv_defaults.cpp): sm_75's is opt-in.
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
}
bool has(const std::vector<std::pair<std::string, std::string>>& t, const char* k, const char* v) {
    for (const auto& kv : t)
        if (kv.first == k && kv.second == v) return true;
    return false;
}
}  // namespace

int main() {
    unsetenv("STRATA_EMULATE_CC");
    unsetenv("STRATA_SM75_STACK");
    check(strata::core::nv_default_env(7, 5).empty(), "sm_75 table empty without STRATA_SM75_STACK");
    setenv("STRATA_SM75_STACK", "0", 1);
    check(strata::core::nv_default_env(7, 5).empty(), "sm_75 table empty with STRATA_SM75_STACK=0");
    setenv("STRATA_SM75_STACK", "1", 1);
    const auto t = strata::core::nv_default_env(7, 5);
    check(has(t, "STRATA_KQ_SPLIT", "1") && has(t, "STRATA_EXPERT_V2K", "1") && has(t, "STRATA_SM75_Q8ROWS1", "1"),
          "sm_75 table has the expert and Q8_0 switches");
    check(strata::core::nv_default_env(7, 0).empty(), "sm_70 untouched by STRATA_SM75_STACK");
    check(!strata::core::nv_default_env(8, 6).empty(), "sm_86 table unchanged");
    unsetenv("STRATA_SM75_STACK");
    std::printf(fails ? "nv_defaults_test: FAIL\n" : "nv_defaults_test: OK\n");
    return fails ? 1 : 0;
}
