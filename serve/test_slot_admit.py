"""STRATA_SLOT_ADMIT: a batch admission is read into its slot beside the decoding slots.

The engine answers a BGEN at once with `BQ <slot> 1` (queued; the rest comes on the slot's own lines: BRP progress,
BRA <slot> <1|0> when it decodes or does not, BRE when refused, then BT/BDONE) or `BQ <slot> 0` (the old way, on the
control lines).  A fake engine speaking both runs behind the real StrataEngine and Service: the server lets go of the
control lines once a read is queued, so the next request's BGEN reaches the engine during the read; the slots go on
decoding meanwhile; a cancel or a refusal frees the slot."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve

FAKE_ADMIT = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1])
STEP, CH = 0.02, 32
lines = queue.Queue()
def reader():
    for l in sys.stdin:
        lines.put(l.strip())
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print(f"INFO engine=0.1.45 batch_slots={slots} slot_cache=1 batch_groups={slots} slot_admit=1", flush=True)
print("READY 8192 stop", flush=True)
log = open(args[args.index("--log") + 1], "a")
def note(s):
    log.write(s + "\n"); log.flush()
def reply(ids):            # the rest of the reply after what the prompt already ends with (a continued request)
    ids = [int(x) for x in ids]
    long_one = b"LONGREPLY" in bytes(x & 255 for x in ids)
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R[max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k]):]
active = {}      # slot -> [tokens left, max_new, produced]
reads = []       # admissions read beside the slots, one at a time: [slot, ids, max_new, done, rows at start]
rows = 0
stopped = set()  # BSTOPped slots: they end "cancel"
def window():
    global rows
    for b in sorted(active):
        left, max_new, produced = active[b]
        t = left.pop(0) if left else 257
        produced += 1
        rows += 1
        print(f"BT {b} {t}", flush=True)
        if t == 257 or produced >= max_new:
            fin = "stop" if t == 257 else "cancel" if b in stopped else "length"
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
def read_step():
    if not reads:
        return
    r = reads[0]
    slot, ids, max_new = r[0], r[1], r[2]
    r[3] = min(len(ids) - 1, r[3] + CH)
    print(f"BRP {slot} {r[3]} {len(ids)} 10 100.0", flush=True)
    if r[3] >= len(ids) - 1:
        reads.pop(0)
        note(f"READ {slot} rows {rows - r[4]}")
        print(f"BRA {slot} 1 prompt={len(ids)} reused=0 read_ms=5.0 wait_ms=1.0", flush=True)
        active[slot] = [reply(ids), max_new, 0]
        if reads:
            reads[0][4] = rows
while True:
    try:
        line = lines.get(timeout=STEP if active or reads else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD") or line == "STOP":
        continue
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        for r in list(reads):
            if r[0] == b:
                reads.remove(r)
                note(f"CANCEL {b}")
                print(f"BRA {b} 0 cancel", flush=True)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        ids = f[-1].split(",")
        pb = bytes(int(x) & 255 for x in ids)
        if f[0] == "GEN":                     # solo: a token a step until STOP
            note(f"GEN {len(ids)}")
            out, halt, held = 0, False, []
            for t in reply(ids)[:int(f[1])]:
                print(f"T {t}", flush=True)
                out += 1
                time.sleep(STEP)
                while True:
                    try:
                        l2 = lines.get_nowait()
                    except queue.Empty:
                        break
                    if l2 == "STOP":
                        halt = True
                    else:
                        held.append(l2)
                if halt:
                    break
            for l2 in held:
                lines.put(l2)
            fin = "cancel" if halt and out < len(reply(ids)) else "stop"
            print(f"DONE {out} {len(ids)} 5.0 1.0 {fin} 0 0 0", flush=True)
            continue
        slot, max_new = int(f[1]), int(f[2])
        direct = bool(active or reads)
        note(f"BGEN {slot} {len(ids)} direct={int(direct)} reading={reads[0][0] if reads else -1}")
        print(f"BQ {slot} {int(direct)}", flush=True)
        if direct:
            if b"REFUSEME" in pb:
                print(f"BRE {slot} the prompt and max_new exceed the context", flush=True)
                continue
            reads.append([slot, ids, max_new, 0, rows])
            continue
        toks = reply(ids)                     # the old way: the first token on the control lines, then the slot
        print(f"T {toks[0]}", flush=True)
        print(f"DONE 1 {len(ids)} 5.0 1.0 length 0 0 0", flush=True)
        active[slot] = [toks[1:], max_new, 1]
        print(f"BADM {slot} 1", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
    read_step()
'''


class SlotAdmit(unittest.TestCase):
    def start(self, slots=3):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_ADMIT, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", ["--batch", str(slots), "--log", str(self.log)])
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=200):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def lines(self):
        return self.log.read_text().splitlines()

    def wait_free(self):
        deadline = time.time() + 10
        while any(self.engine.slot_busy) and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(any(self.engine.slot_busy))

    def test_engine_says_so(self):
        self.start()
        self.assertTrue(self.engine.slot_admit)

    def test_reads_beside_the_slots_and_the_next_bgen_is_not_held_back(self):
        """A decodes; B's long prompt is read beside it (BQ 1); C's BGEN reaches the engine while B is still being
        read: the server gave the control lines back at B's BQ.  A decoded during B's read, every reply is whole."""
        self.start()
        out, errors = {}, []

        def go(k, text):
            try:
                out[k] = self.chat(text)
            except Exception as e:      # noqa: BLE001 - reported below
                errors.append((k, repr(e)))
        texts = {"a": "LONGREPLY first", "b": "long " * 600 + "LONGREPLY", "c": "short third"}
        threads = []
        for k in ("a", "b", "c"):
            th = threading.Thread(target=go, args=(k, texts[k]))
            th.start()
            threads.append(th)
            time.sleep(0.15)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(out["a"], "ok, " + "la " * 15 + "done.")
        self.assertEqual(out["b"], "ok, " + "la " * 15 + "done.")
        self.assertEqual(out["c"], "ok, done.")
        log = self.lines()
        bgens = [x.split() for x in log if x.startswith("BGEN ")]
        long_b = next(x for x in bgens if int(x[2]) > 2000)
        self.assertEqual(long_b[3], "direct=1", log)
        short_c = next(x for x in bgens if int(x[2]) < 200 and x is not bgens[0])
        self.assertEqual(short_c[4], f"reading={long_b[1]}", log)        # sent while B was being read
        read_b = next(x.split() for x in log if x.startswith(f"READ {long_b[1]} "))
        self.assertGreater(int(read_b[3]), 0, log)                       # the others decoded during the read
        self.wait_free()
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})

    def test_a_cancelled_queued_admission_frees_its_slot(self):
        self.start()
        e = self.engine
        bg = threading.Thread(target=lambda: self.chat("LONGREPLY background"))
        bg.start()
        time.sleep(0.15)
        cancel = threading.Event()
        got = []
        th = threading.Thread(target=lambda: got.extend(e.generate_batched(list(b"long " * 600), 64, {}, cancel)))
        th.start()
        deadline = time.time() + 5
        while not any(x.startswith("BGEN ") and "direct=1" in x for x in self.lines()) and time.time() < deadline:
            time.sleep(0.01)
        cancel.set()
        th.join(10)
        self.assertFalse(th.is_alive())
        bg.join(30)
        self.assertTrue(any(x.startswith("CANCEL ") for x in self.lines()), self.lines())
        self.assertEqual([t for t in got if t is not None], [])
        self.wait_free()
        self.assertEqual(self.chat("after"), "ok, done.")

    def test_a_refused_admission_is_an_error_and_frees_its_slot(self):
        self.start()
        bg = threading.Thread(target=lambda: self.chat("LONGREPLY background"))
        bg.start()
        time.sleep(0.15)
        with self.assertRaises(urllib.error.HTTPError):
            self.chat("REFUSEME please")
        bg.join(30)
        self.wait_free()
        self.assertEqual(self.chat("after"), "ok, done.")

    def test_control_line_bq(self):
        """_control returns at `BQ <slot> 1`; `BQ <slot> 0` is read past (the old way's lines follow)."""
        e = StrataEngine("missing-executable", [], lazy=True)
        import queue
        e.lines = queue.Queue()
        e._ctl_mode = "batch"
        for line in ("BQ 2 0", "T 5", "DONE 1 9 1.0 1.0 length 0 0 0", "BADM 2 1"):
            e.lines.put(line)
        toks = []
        list(e._control(threading.Event(), toks.append))
        self.assertEqual((toks, e._ctl_result), ([5], ("badm", True)))
        e.lines.put("BQ 1 1")
        list(e._control(threading.Event(), toks.append))
        self.assertEqual(e._ctl_result, ("queued", True))


class PickSlotAdmit(unittest.TestCase):
    def test_no_hole_below_a_busy_slot(self):
        """Pipelined groups of 2 (slots {0,1}, {2,3}): with STRATA_SLOT_ADMIT a slot below a busy one would make its
        group wait for the whole read, so a slot above the busy ones comes first; without, the hole first, as before."""
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch, e.slot_gs, e.slot_groups = 4, 2, 2
        e.slot_order = [0, 2, 1, 3]
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy = [False, True, False, False]
        e.slot_held = [[] for _ in range(4)]
        e.slot_used = [0.0] * 4
        e.slot_admit = False
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)
        e.slot_admit = True
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
