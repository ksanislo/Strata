"""serve/test_engine_stats.py - GET /metrics' engine-wide figures ("aggregate", "conversations"): the stdout counts and
their 10 s / 60 s rates, the engine log's figures, and the whole thing over the fake batch engine (no GPU).

    python -m unittest serve.test_engine_stats -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.engine_stats import EngineLog, EngineStats  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Counts(unittest.TestCase):
    def test_tokens_reads_reuse_and_errors(self):
        clk = Clock()
        es = EngineStats(clock=clk, wall=lambda: 5.0)
        for _ in range(3):
            es.on_line("T 17\n")
        es.on_line("BT 0 5\n")
        es.on_line("BT 3 9\n")
        self.assertEqual(es.generated, 5)
        # a solo read: its first PP line counts what it read so far (rate x time), the next ones the position's step
        es.on_line("PP 3000 5000 1000 2000.0\n")           # resumed at 1000 (reused), 2000 fresh tokens in 1 s
        es.on_line("PP 5000 5000 2000 2000.0\n")
        self.assertEqual(es.prompt_read, 4000)
        es.on_line("DONE 4 5000 2000 100 stop 0 0 1000\n")  # reused field
        self.assertEqual(es.reused, 1000)
        # a read beside the slots (BRP per slot), and one that gave way on the control lines and goes on in slot 2
        es.on_line("BRP 1 2048 8000 500 4096.0\n")
        es.on_line("BRP 1 4096 8000 1000 4096.0\n")
        es.on_line("PP 1024 9000 300 3413.3\n")
        es.on_line("BRP 2 2048 9000 250 4096.0\n")        # a new read for slot 2: 1024 fresh in 0.25 s
        self.assertEqual(es.prompt_read, 4000 + 4096 + 1024 + 1024)
        es.on_line("BRA 1 1 prompt=8000 reused=0 read_ms=1000 wait_ms=0\n")
        es.on_line("BRA 2 1 prompt=9000 reused=512 read_ms=1000 wait_ms=0\n")
        self.assertEqual(es.reused, 1512)
        es.on_line("RESUME 700\n")                         # a solo read that says what it reuses first
        es.on_line("PP 1000 3000 10 1.0\n")
        self.assertEqual(es.prompt_read, 4000 + 4096 + 1024 + 1024 + 300)
        es.on_line("ERR verify: something\n")
        t = es.totals()
        self.assertEqual((t["errors"], t["last_err"]), (1, "verify: something"))
        self.assertEqual(t["reads"], 5)
        es.on_line("INFO sess_json={}\n")                   # other lines change nothing
        self.assertEqual(es.generated, 5)

    def test_rates_over_10_and_60_s(self):
        clk = Clock()
        es = EngineStats(clock=clk)
        es.on_line("INFO x\n")                             # the first sample, at t0
        for _ in range(120):                               # 120 s: 10 tokens/s, 100 prompt tokens/s
            clk.t += 0.25
            es.on_line("T 1\n")
            clk.t += 0.25
            es.on_line("T 1\n")
            clk.t += 0.25
            es.on_line("T 1\n")
            clk.t += 0.25
            es.on_line("BT 0 1\n")
        r = es.rates()
        self.assertAlmostEqual(r["generated_tok_s"]["10s"], 4.0, delta=0.3)
        self.assertAlmostEqual(r["generated_tok_s"]["60s"], 4.0, delta=0.2)
        self.assertEqual(r["prompt_read_tok_s"]["10s"], 0.0)
        # idle: the 10 s window empties, the 60 s one keeps what came in it
        clk.t += 30
        r = es.rates()
        self.assertEqual(r["generated_tok_s"]["10s"], 0.0)
        self.assertAlmostEqual(r["generated_tok_s"]["60s"], 2.0, delta=0.2)
        clk.t += 61
        self.assertEqual(es.rates()["generated_tok_s"]["60s"], 0.0)

    def test_rates_right_after_load_cover_the_time_since(self):
        clk = Clock()
        es = EngineStats(clock=clk)
        clk.t += 2
        for _ in range(20):
            es.on_line("T 1\n")
        self.assertAlmostEqual(es.rates()["generated_tok_s"]["10s"], 10.0, delta=0.1)

    def test_a_reader_while_the_pump_appends(self):
        es = EngineStats()
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                es.on_line("T 1\n")
        th = threading.Thread(target=pump)
        th.start()
        try:
            for _ in range(200):
                es.rates()
        finally:
            stop.set()
            th.join()
        self.assertGreater(es.generated, 0)


LOG = """\
strata serve: 430 MiB of VRAM free with everything loaded
strata serve: CUDA2: 251 MiB of VRAM free with everything loaded - LOW: its window graphs are captured on first use
strata verify: capturing the 4-token window (CUDA1: 518 MiB of VRAM free)
strata verify: captured the 4-token window (upload no error, sync no error)
strata verify: captured the batch window over slots 0,1,2
strata verify: no VRAM for a new window graph: freed the graphs of 2 older batch slot layouts, 1 kept (a larger --vram-reserve-mib keeps more)
strata batch: CUDA0: 372 MiB of VRAM free, below its floor of 544 MiB (STRATA_VRAM_FLOOR_MIB; 0 of 24 batch layouts held, 0.0 MiB a graph pair): 172 MiB of expert cache / slot budget given back
strata batch (pipelined, 4 groups of 2): 30 group-steps, 82 rows in 7323 ms = 11.2 rows/s (admissions included)
strata batch (pipelined): MTP proposals accepted 91 of 114 (79.8%); 58 windows of 2.97 rows avg, 188.6 ms launch to picks, drafts 6.0 ms after; 2.57 tokens per window
strata serve: decode expert cache hit rate: 65.2% (13967 hits / 21415 lookups); 2105 more read by the GPU over PCIe or from another GPU (8.9% of all 23520 routed)
strata serve: decode expert cache hit rate: 96.9% (465 hits / 480 lookups)
strata batch: read 6220 tokens on a thread of its own in 11914 ms (concurrent, to 6220 of 30000); the slots decoded 12 rows meanwhile (1.0 rows/s; stage 1: 0 windows, 0.0 ms host each)
strata batch: 4 admissions read beside the slots in 34687 ms (queued, STRATA_SLOT_ADMIT_MULTI); the groups stopped 1 ms for the loan (lend, refill) and decoded 113 rows (stage 1: 42 windows, pool 29.0 ms each)
strata batch: 5 admissions read beside the slots in 54666 ms (queued, STRATA_SLOT_ADMIT_MULTI); yielded with 3 held, the loan kept; decoded 116 rows (stage 1: 44 windows, pool 24.9 ms each)
strata batch: STRATA_READ_YIELD: a text request waits: the reads beside the slots stop at their next piece boundary (1 piece taken back)
strata batch: STRATA_READ_YIELD: a text request waits: the reads beside the slots stop at their next piece boundary (0 pieces taken back)
strata batch: STRATA_READ_YIELD: 3 reads held at a piece boundary (loan kept): slot 2 at 16384/18344 slot 4 at 8192/20498
"""


class Log(unittest.TestCase):
    def test_the_engine_log_figures(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "engine.log")
            Path(path).write_text("strata serve: decode expert cache hit rate: 1.0% (1 hits / 100 lookups)\n")
            start = os.path.getsize(path)                   # an earlier run's lines are not this one's
            with open(path, "a") as f:
                f.write(LOG + "partial line without its end")
            el = EngineLog(wall=lambda: 7.0)
            v = el.poll(path, start)
            c, lt = v["counts"], v["latest"]
            self.assertEqual(c["graph_captures"], 2)
            self.assertEqual((c["no_vram_for_graph"], c["graphs_freed"]), (1, 2))
            self.assertEqual(c["vram_floor_givebacks"], 1)
            self.assertEqual(lt["vram_free_mib"]["GPU"]["mib"], 430)
            self.assertEqual(lt["vram_free_mib"]["CUDA2"]["mib"], 251)
            self.assertEqual(lt["vram_free_mib"]["CUDA1"], {"mib": 518, "what": "graph capture", "at": 7.0})
            self.assertEqual(lt["vram_free_mib"]["CUDA0"]["what"], "below floor")
            self.assertEqual(lt["pipelined"]["rows_s"], 11.2)
            self.assertEqual((lt["pipelined"]["groups"], lt["pipelined"]["group_slots"]), (4, 2))
            self.assertEqual(lt["pipelined_mtp"]["accept_pct"], 79.8)
            self.assertEqual(lt["hit_rate"]["pct"], 96.9)
            self.assertEqual(c["lookups"], 21415 + 480)
            self.assertAlmostEqual(v["hit_rate_pct_since_start"], 100 * (13967 + 465) / (21415 + 480), places=1)
            self.assertEqual(lt["thread_read"]["tokens"], 6220)
            self.assertEqual((c["beside_reports"], c["beside_admissions"]), (2, 9))
            self.assertEqual(lt["beside"]["held"], 3)
            self.assertEqual((c["read_yields"], c["pieces_taken_back"]), (2, 1))
            self.assertEqual(lt["reads_held"], 3)
            with open(path, "a") as f:                       # the partial line ends; more comes
                f.write("\nstrata batch: STRATA_READ_YIELD: 3 held reads go on\n")
            v = el.poll(path, start)
            self.assertEqual(v["latest"]["reads_held"], 0)
            self.assertEqual(v["counts"]["held_reads_resumed"], 3)
            self.assertEqual(v["counts"]["graph_captures"], 2)      # nothing counted twice
            v = el.poll(path, start + 1)                             # another engine start: counts start again
            self.assertEqual(v["counts"]["graph_captures"], 2)
            v = el.poll(path, os.path.getsize(path))
            self.assertEqual(v["counts"]["graph_captures"], 0)


class OverTheFakeEngine(unittest.TestCase):
    """The real StrataEngine and Service over the fake batch engine of test_parallel."""

    def setUp(self):
        from serve.test_parallel import ParallelService
        self.ps = ParallelService("test_concurrent_requests_share_the_slots")

    def tearDown(self):
        self.ps.tearDown()

    def test_aggregate_and_conversations(self):
        ps = self.ps
        ps.start(2)
        texts = ["first LONGREPLY", "second LONGREPLY", "third waits for a slot " + "and reads a longer prompt " * 8]
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = ps.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            seen.append(ps.get("/metrics"))
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        agg = [m["aggregate"] for m in seen]
        self.assertTrue(all(a["parallel"] == 2 and sum(a["slots"].values()) == 2 for a in agg))
        # two conversations at once, each its own row with its own figures
        rows = [m["conversations"] for m in seen]
        both = [r for r in rows if sum(1 for c in r if c["state"] == "decoding") >= 2]
        self.assertTrue(both, "two conversations decoding at once were not seen")
        self.assertEqual(len({c["slot"] for c in both[0] if c["state"] == "decoding"}), 2)
        self.assertTrue(all(c["generated"] >= 0 and c["prompt_tokens"] > 0 for r in rows for c in r))
        self.assertTrue(any(c["ttft_s"] is not None and c["tok_s"] is not None for r in rows for c in r))
        self.assertTrue(any(a["generated_tok_s"]["10s"] > 0 for a in agg))
        final = ps.svc.metrics()
        a = final["aggregate"]
        self.assertEqual(final["conversations"], [])
        self.assertEqual(a["queue"]["waiting"], 0)
        # every token the three replies streamed was counted by the engine-wide counter
        self.assertGreaterEqual(a["totals"]["generated"], sum(r["usage"]["completion_tokens"] for r in out.values()))
        self.assertGreater(a["totals"]["prompt_read"] + a["totals"]["reused"], 0)
        self.assertEqual(a["slots"]["reading"] + a["slots"]["decoding"], 0)


if __name__ == "__main__":
    unittest.main()
