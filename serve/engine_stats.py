"""serve/engine_stats.py - engine-wide figures for GET /metrics ("aggregate"): what all batch slots do together.

EngineStats counts the engine's stdout lines on the pump thread (its only writer, so no lock): generated tokens (T and
BT lines), prompt tokens read (PP and BRP progress: the solo / control-line reads and the reads beside the slots),
reused tokens (DONE, BRA) and ERR lines.  Every SAMPLE_S it keeps the running totals in a ring, from which the
10 s / 60 s rates are taken.  EngineLog reads the engine's stderr (its log) on from where it last read and keeps the
latest figures it reports there (expert cache hit rate, pipelined rows/s, reads beside the slots, read yields, VRAM free,
window graph captures)."""
from __future__ import annotations

import collections
import os
import re
import time

SAMPLE_S = 0.25         # ring resolution: a rate is off by at most this much of its window
RING = 320              # 80 s of samples, enough for the 60 s window
WINDOWS = (10, 60)


class EngineStats:
    def __init__(self, clock=time.monotonic, wall=time.time):
        self.clock = clock
        self.since = wall()
        self.t0 = clock()
        self.generated = 0          # tokens the engine sent (solo T + slot BT), drafts included as they come out
        self.prompt_read = 0        # prompt tokens read (not reused), as the PP / BRP progress lines report them
        self.reused = 0             # prompt tokens not read again (DONE's reused field, BRA reused=)
        self.reads = 0              # reads that reported progress (a read that gave way and went on counts twice)
        self.errors = 0
        self.last_err = None
        self.last_err_at = None
        self._progress: dict = {}   # read key ("ctl" or slot) -> (total, ms, position) of its last progress line
        self._ring = collections.deque(maxlen=RING)
        self._last_sample = -1e9

    def on_line(self, line: str) -> None:
        """One stdout line (the pump thread)."""
        c = line[:1]
        if c == "T" and line.startswith("T "):
            self.generated += 1
        elif c == "B":
            if line.startswith("BT "):
                self.generated += 1
            elif line.startswith("BRP "):
                f = line.split()
                if len(f) >= 6:
                    self._read(f[1], f[2], f[3], f[4], f[5])
            elif line.startswith("BRA "):
                for kv in line.split()[3:]:
                    if kv.startswith("reused="):
                        self.reused += _int(kv[7:])
                self._progress.pop(line.split()[1] if len(line.split()) > 1 else None, None)
        elif c == "P" and line.startswith("PP "):
            f = line.split()
            if len(f) >= 5:
                self._read("ctl", f[1], f[2], f[3], f[4])
        elif c == "R" and line.startswith("RESUME "):   # before a solo read: this many prompt tokens are reused
            self._progress["ctl"] = ("resume", 0.0, _int(line.split()[1]))
        elif c == "D" and line.startswith("DONE"):
            f = line.split()
            if len(f) >= 9:
                self.reused += _int(f[8])
            self._progress.pop("ctl", None)
        elif c == "E" and line.startswith("ERR"):
            self.errors += 1
            self.last_err = line[4:].strip()[:300]
            self.last_err_at = time.time()
        now = self.clock()
        if now - self._last_sample >= SAMPLE_S:
            self._last_sample = now
            self._ring.append((now, self.generated, self.prompt_read, self.reused))

    def _read(self, key, pos, total, ms, rate) -> None:
        try:
            pos, total, ms, rate = int(pos), int(total), float(ms), float(rate)
        except ValueError:
            return
        prev = self._progress.get(key)
        if prev is not None and prev[0] == "resume" and pos >= prev[2]:
            fresh = pos - prev[2]                       # a solo read's first line: past the reused tokens
            self.reads += 1
        elif prev is not None and prev[0] == total and ms >= prev[1] and pos >= prev[2]:
            fresh = pos - prev[2]                       # the same read went on
        else:                                           # a new read (or one going on after it gave way): its fresh
            fresh = min(pos, max(0, round(rate * ms / 1000.0)))   # tokens so far are its rate times its time
            self.reads += 1
        self._progress[key] = (total, ms, pos)
        self.prompt_read += fresh

    def _samples(self) -> list:
        for _ in range(3):
            try:
                return list(self._ring)
            except RuntimeError:                        # appended to while copied
                continue
        return []

    def rates(self, now: float | None = None) -> dict:
        """tok/s of generated, prompt-read and reused tokens over the last 10 s and 60 s (0 when idle)."""
        now = self.clock() if now is None else now
        cur = (self.generated, self.prompt_read, self.reused)
        ring = self._samples()
        out = {}
        for w in WINDOWS:
            start = now - w
            base = (self.t0, 0, 0, 0)
            for s in ring:                              # the last sample at or before the window's start
                if s[0] <= start:
                    base = s
                else:
                    break
            if ring and ring[-1][0] <= start and cur == tuple(ring[-1][1:]):
                out[w] = (0.0, 0.0, 0.0)                # nothing since well before the window
                continue
            dt = max(now - base[0], 1e-6)
            if dt < 0.5:
                out[w] = (None, None, None)
                continue
            out[w] = tuple(round(max(0, cur[i] - base[i + 1]) / dt, 1) for i in range(3))
        return {"generated_tok_s": {f"{w}s": out[w][0] for w in WINDOWS},
                "prompt_read_tok_s": {f"{w}s": out[w][1] for w in WINDOWS},
                "reused_tok_s": {f"{w}s": out[w][2] for w in WINDOWS}}

    def totals(self) -> dict:
        return {"since": self.since, "generated": self.generated, "prompt_read": self.prompt_read,
                "reused": self.reused, "reads": self.reads, "errors": self.errors,
                "last_err": self.last_err, "last_err_at": self.last_err_at}


def _int(s, default=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _num(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


class EngineLog:
    """The engine's own stderr figures, from its log: read on from where it was last read, from the start of the
    engine's current run (as ConvCacheLog).  Counts are since that start; `latest` holds each figure's last value."""
    READ_MAX = 1 << 20

    HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)"
                     r"(?:; (\d+) more read by the GPU over PCIe.*?\(([\d.]+)% of all)?")
    PIPE = re.compile(r"strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, (\d+) rows in ([\d.]+) ms"
                      r" = ([\d.]+) rows/s")
    PIPE_MTP = re.compile(r"strata batch \(pipelined\): MTP proposals accepted (\d+) of (\d+) \(([\d.]+)%\); (\d+) windows"
                          r" of ([\d.]+) rows avg, ([\d.]+) ms launch to picks")
    THREAD = re.compile(r"read (\d+) tokens on a thread of its own in ([\d.]+) ms \(.*?to (\d+) of (\d+)\); the slots "
                        r"decoded (\d+) rows meanwhile \(([\d.]+) rows/s")
    BESIDE = re.compile(r"(\d+) admissions? read beside the slots in ([\d.]+) ms.*?(?:yielded with (\d+) held.*?)?"
                        r"decoded (\d+) rows")
    STOPPED = re.compile(r"the groups stopped (\d+) ms for the loan")
    Y_WAIT = re.compile(r"STRATA_READ_YIELD: a text request waits.*?\((\d+) pieces? taken back\)")
    Y_HELD = re.compile(r"STRATA_READ_YIELD: (\d+) reads? held at a piece boundary")
    Y_GO = re.compile(r"STRATA_READ_YIELD: (\d+) held reads? go on")
    NOGRAPH = re.compile(r"no VRAM for a new window graph: freed the graphs of (\d+) older")
    FREE = re.compile(r"(?:(CUDA\d+|HIP\d+|GPU\d+): )?(\d+) MiB of VRAM free")

    def __init__(self, wall=time.time):
        self.wall = wall
        self.key, self.pos = None, 0
        self.reset()

    def reset(self):
        self.counts = {"pipelined_reports": 0, "pipelined_rows": 0, "pipelined_ms": 0.0,
                       "thread_reads": 0, "thread_read_tokens": 0, "thread_read_ms": 0.0,
                       "beside_reports": 0, "beside_admissions": 0, "beside_ms": 0.0, "beside_loan_stop_ms": 0,
                       "read_yields": 0, "pieces_taken_back": 0, "held_reads_resumed": 0,
                       "graph_captures": 0, "no_vram_for_graph": 0, "graphs_freed": 0,
                       "vram_floor_givebacks": 0, "vram_low_warnings": 0,
                       "hits": 0, "lookups": 0}
        self.latest = {"hit_rate": None, "pipelined": None, "pipelined_mtp": None, "thread_read": None,
                       "beside": None, "reads_held": 0, "reads_held_at": None, "no_vram_for_graph_at": None,
                       "vram_free_mib": {}}

    def poll(self, path, start) -> dict:
        if path and start is not None:
            if self.key != (path, start):               # another start of the engine: its figures start again
                self.key, self.pos = (path, start), start
                self.reset()
            try:
                size = os.path.getsize(path)
                if size < self.pos:
                    self.pos = 0
                if size - self.pos > self.READ_MAX:
                    self.pos = size - self.READ_MAX
                with open(path, "rb") as f:
                    f.seek(self.pos)
                    data = f.read(size - self.pos)
            except OSError:
                data = b""
            end = data.rfind(b"\n") + 1
            self.pos += end
            for line in data[:end].decode("utf-8", "replace").splitlines():
                self.feed(line)
        c = self.counts
        hit_all = round(100.0 * c["hits"] / c["lookups"], 1) if c["lookups"] else None
        return {"counts": dict(c), "hit_rate_pct_since_start": hit_all,
                "latest": {**self.latest, "vram_free_mib": dict(self.latest["vram_free_mib"])}}

    def feed(self, line: str) -> None:
        if "strata" not in line and "STRATA" not in line:
            return
        now = self.wall()
        c, lt = self.counts, self.latest
        if "decode expert cache hit rate" in line:
            m = self.HIT.search(line)
            if m:
                c["hits"] += int(m.group(2))
                c["lookups"] += int(m.group(3))
                lt["hit_rate"] = {"pct": float(m.group(1)), "hits": int(m.group(2)), "lookups": int(m.group(3)),
                                  "pcie_pct": _num(m.group(5)), "at": now}
            return
        if "strata batch (pipelined" in line:
            m = self.PIPE.search(line)
            if m:
                c["pipelined_reports"] += 1
                c["pipelined_rows"] += int(m.group(4))
                c["pipelined_ms"] += float(m.group(5))
                lt["pipelined"] = {"groups": int(m.group(1)), "group_slots": int(m.group(2)),
                                   "group_steps": int(m.group(3)), "rows": int(m.group(4)), "ms": float(m.group(5)),
                                   "rows_s": float(m.group(6)), "at": now}
                return
            m = self.PIPE_MTP.search(line)
            if m:
                lt["pipelined_mtp"] = {"accepted": int(m.group(1)), "proposed": int(m.group(2)),
                                       "accept_pct": float(m.group(3)), "windows": int(m.group(4)),
                                       "rows_avg": float(m.group(5)), "launch_to_picks_ms": float(m.group(6)), "at": now}
            return
        if "on a thread of its own" in line:
            m = self.THREAD.search(line)
            if m:
                c["thread_reads"] += 1
                c["thread_read_tokens"] += int(m.group(1))
                c["thread_read_ms"] += float(m.group(2))
                lt["thread_read"] = {"tokens": int(m.group(1)), "ms": float(m.group(2)), "to": int(m.group(3)),
                                     "of": int(m.group(4)), "slot_rows": int(m.group(5)),
                                     "slot_rows_s": float(m.group(6)), "at": now}
            return
        if "read beside the slots in" in line:
            m = self.BESIDE.search(line)
            if m:
                stop = self.STOPPED.search(line)
                c["beside_reports"] += 1
                c["beside_admissions"] += int(m.group(1))
                c["beside_ms"] += float(m.group(2))
                c["beside_loan_stop_ms"] += int(stop.group(1)) if stop else 0
                lt["beside"] = {"admissions": int(m.group(1)), "ms": float(m.group(2)),
                                "held": int(m.group(3)) if m.group(3) else 0, "decoded_rows": int(m.group(4)),
                                "loan_stop_ms": int(stop.group(1)) if stop else None, "at": now}
            return
        if "STRATA_READ_YIELD" in line:
            m = self.Y_WAIT.search(line)
            if m:
                c["read_yields"] += 1
                c["pieces_taken_back"] += int(m.group(1))
                return
            m = self.Y_HELD.search(line)
            if m:
                lt["reads_held"], lt["reads_held_at"] = int(m.group(1)), now
                return
            m = self.Y_GO.search(line)
            if m:
                c["held_reads_resumed"] += int(m.group(1))
                lt["reads_held"], lt["reads_held_at"] = 0, now
            return
        if "no VRAM for a new window graph" in line:
            m = self.NOGRAPH.search(line)
            c["no_vram_for_graph"] += 1
            c["graphs_freed"] += int(m.group(1)) if m else 0
            lt["no_vram_for_graph_at"] = now
            return
        if "strata verify: captured the" in line:
            c["graph_captures"] += 1
            return
        if "MiB of VRAM free" in line:
            m = self.FREE.search(line)
            if m:
                card = m.group(1) or "GPU"
                what = ("graph capture" if "capturing" in line else "below floor" if "below its floor" in line
                        else "request start" if "at the start of this request" in line
                        else "loaded" if "everything loaded" in line else "reported")
                lt["vram_free_mib"][card] = {"mib": int(m.group(2)), "what": what, "at": now}
                if "below its floor" in line:
                    c["vram_floor_givebacks"] += 1
                elif "WARNING: only" in line:
                    c["vram_low_warnings"] += 1
