"""A DONE / BDONE that says `vram` fails that request the #879 way (sent once more when nothing went out), not the
engine; and the engine's stdout ERR lines reach its log."""
import queue
import tempfile
import threading
import unittest
from pathlib import Path

from serve.server import EngineNonFinite, EngineOutOfVram, StrataEngine, raise_bad_finish


class BadFinish(unittest.TestCase):
    def test_vram_is_its_own_error(self):
        with self.assertRaises(EngineOutOfVram) as c:
            raise_bad_finish("vram")
        self.assertIsInstance(c.exception, EngineNonFinite)   # Service.run's retry-once path
        self.assertIn("no free memory", str(c.exception))

    def test_nonfinite_unchanged(self):
        with self.assertRaises(EngineNonFinite) as c:
            raise_bad_finish("nonfinite")
        self.assertNotIsInstance(c.exception, EngineOutOfVram)

    def test_ordinary_finishes_pass(self):
        for f in ("stop", "length", "cancel", None):
            raise_bad_finish(f)


class ErrLinesLogged(unittest.TestCase):
    def test_err_line_goes_to_the_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "engine.log"
            eng = StrataEngine.__new__(StrataEngine)

            class Proc:
                stdout = iter(["T 5\n", "ERR verify: batch instantiate: out of memory (CUDA3, 2 MiB free)\n"])

            eng.proc, eng.lines, eng.slot_q = Proc(), queue.Queue(), []
            eng.log = open(p, "a", encoding="utf-8")
            t = threading.Thread(target=eng._pump)
            t.start()
            t.join(5)
            eng.log.close()
            text = p.read_text()
            self.assertIn("[strata] engine said: ERR verify: batch instantiate: out of memory (CUDA3, 2 MiB free)", text)
            self.assertNotIn("T 5", text)
            self.assertEqual(eng.last_err, "verify: batch instantiate: out of memory (CUDA3, 2 MiB free)")


if __name__ == "__main__":
    unittest.main()
