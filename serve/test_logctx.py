"""serve/logctx.py: timestamped lines with request tags, the client's address and kind, conversation keys, events."""
import io
import json
import os
import tempfile
import threading
import time
import unittest

from serve import logctx


class Lines(unittest.TestCase):
    def test_a_line_gets_the_time_and_its_requests_tags(self):
        out = io.StringIO()
        s = logctx.StampedStream(out)
        logctx.begin("ab12", "7c21b0")
        logctx.set_slot(4)
        try:
            print("[strata] done: 5 tokens", file=s)
        finally:
            logctx.end()
        print("[strata] idle", file=s)
        a, b = out.getvalue().splitlines()
        self.assertRegex(a, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[req ab12 conv 7c21b0 slot 4\] \[strata\] done: 5 tokens$")
        self.assertRegex(b, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[strata\] idle$")
        self.assertEqual(logctx.unstamp(b), "[strata] idle")

    def test_threads_never_split_each_others_lines(self):
        out = io.StringIO()
        s = logctx.StampedStream(out)

        def burst(tag):
            for i in range(300):
                print(f"{tag} line {i} end", file=s)
        ts = [threading.Thread(target=burst, args=(t,)) for t in "abcd"]
        [t.start() for t in ts]
        [t.join() for t in ts]
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1200)
        for ln in lines:
            self.assertRegex(logctx.unstamp(ln), r"^[abcd] line \d+ end$")

    def test_the_engine_log_gets_milliseconds_and_blank_lines_stay_blank(self):
        f = io.StringIO()
        log = logctx.StampedLog(f)
        r, w = os.pipe()
        with os.fdopen(w, "w") as wf:
            wf.write("strata serve: one\n\nstrata batch: two\n")
        with os.fdopen(r, "r") as rf:
            logctx.copy_lines(rf, log)
        a, blank, b = f.getvalue().split("\n")[:3]
        self.assertRegex(a, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} strata serve: one$")
        self.assertEqual(blank, "")
        self.assertEqual(logctx.unstamp(b), "strata batch: two")


class Client(unittest.TestCase):
    nets = logctx.trusted_nets(["127.0.0.1", "::1", "172.18.0.0/16"])

    def test_the_address_comes_from_past_the_trusted_proxies(self):
        # llama-swap (127.0.0.1) <- nginx (172.18.0.9) <- the client
        self.assertEqual(logctx.client_ip("127.0.0.1", "203.0.113.7, 172.18.0.9", self.nets), "203.0.113.7")

    def test_an_address_the_client_wrote_itself_is_not_taken(self):
        self.assertEqual(logctx.client_ip("127.0.0.1", "10.9.9.9, 203.0.113.7, 172.18.0.9", self.nets), "203.0.113.7")

    def test_no_trusted_proxy_no_header(self):
        self.assertEqual(logctx.client_ip("203.0.113.7", "10.9.9.9", self.nets), "203.0.113.7")
        self.assertEqual(logctx.client_ip("127.0.0.1", "10.9.9.9", []), "127.0.0.1")

    def test_kinds_and_key_prints(self):
        self.assertEqual(logctx.client_kind("opencode/1.18 ai-sdk", None), "opencode")
        self.assertEqual(logctx.client_kind("Mozilla/5.0 Safari", "https://llm.x/upstream/Q8/"), "strata-web")
        self.assertEqual(logctx.client_kind("Mozilla/5.0", "https://llm.x/ui/models"), "llama-swap-ui")
        self.assertEqual(logctx.client_kind("Python-urllib/3.13", None), "python-urllib")
        k = logctx.key_print("Bearer sk-secret")
        self.assertEqual(len(k), 6)
        self.assertNotIn("secret", k)
        self.assertEqual(k, logctx.key_print("Bearer sk-secret"))
        self.assertIsNone(logctx.key_print(None))

    def test_a_bad_trusted_entry_stops_the_start(self):
        with self.assertRaises(ValueError):
            logctx.trusted_nets(["not-an-address"])


class Conversation(unittest.TestCase):
    A = [9, 9]                                   # the assistant turn's opening

    def test_every_turn_of_a_conversation_has_its_key(self):
        first = [1, 2, 3, 4] + self.A
        later = [1, 2, 3, 4] + self.A + [7, 7, 5, 6] + self.A
        other = [1, 2, 3, 5] + self.A
        self.assertEqual(logctx.conv_key(first, self.A), logctx.conv_key(later, self.A))
        self.assertNotEqual(logctx.conv_key(first, self.A), logctx.conv_key(other, self.A))
        self.assertIsNone(logctx.conv_key([], self.A))


class Events(unittest.TestCase):
    def test_a_file_a_day_and_old_ones_go(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, "M-2000-01-01.jsonl")
            open(old, "w").close()
            os.utime(old, (time.time() - 40 * 86400,) * 2)
            ev = logctx.EventLog(d, "M", days=30)
            ev.write("request", id="x", conv="c", gap_s=None, prompt=12)
            self.assertFalse(os.path.exists(old))
            day = os.path.join(d, f"M-{time.strftime('%Y-%m-%d')}.jsonl")
            rec = json.loads(open(day).read())
            self.assertEqual((rec["ev"], rec["id"], rec["prompt"]), ("request", "x", 12))
            self.assertNotIn("gap_s", rec)        # (None fields are left out)


if __name__ == "__main__":
    unittest.main()
