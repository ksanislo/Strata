"""Logging for a server that runs several conversations at once: when, which request, which conversation, which client.

- Every line the server prints, and every line of the engine's log, starts with the date and time ("log_time" in the
  config, default on; STRATA_LOG_TIME=0 turns it off).  A line printed for a request also names it, its conversation
  and its slot: `2026-10-22 15:41:07 [req 3f9a conv 7c21b0 slot 4] [strata] done: ...`.
- The conversation key is a hash of the prompt up to the first assistant turn (the system prompt and the first user
  message), the same for every turn of one conversation.  No prompt text is logged.
- The client: its address from X-Forwarded-For when the connection comes from a proxy listed in "trusted_proxies"
  (none by default), the kind of client from its User-Agent, and a short fingerprint of its API key (never the key).
- "event_log" (a folder): one JSON line per finished request and per conversation parked or restored, in a file per day
  (<model>-YYYY-MM-DD.jsonl), kept "event_log_days" days (30).
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import struct
import threading
import time

ENABLED = os.environ.get("STRATA_LOG_TIME", "1") != "0"
_CTX = threading.local()
_STAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)? ")


def stamp(ms: bool = False) -> str:
    t = time.time()
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return s + ".%03d" % int((t % 1) * 1000) if ms else s


def unstamp(line: str) -> str:
    """A log line without the time in front (for the readers that look at how a line starts)."""
    return _STAMP.sub("", line, count=1)


# ---- the request a thread works for -------------------------------------------------------------------------------

def begin(req: str, conv: str | None) -> None:
    _CTX.req, _CTX.conv, _CTX.slot = req, conv, None


def set_slot(slot: int | None) -> None:
    if getattr(_CTX, "req", None) is not None:
        _CTX.slot = slot


def set_client(client: dict | None) -> None:
    _CTX.client = client


def end() -> None:
    _CTX.req = _CTX.conv = _CTX.slot = None


def current() -> dict:
    return {"req": getattr(_CTX, "req", None), "conv": getattr(_CTX, "conv", None),
            "slot": getattr(_CTX, "slot", None), "client": getattr(_CTX, "client", None)}


def tags() -> str:
    req = getattr(_CTX, "req", None)
    if req is None:
        return ""
    t = f"req {req}"
    conv = getattr(_CTX, "conv", None)
    if conv:
        t += f" conv {conv}"
    slot = getattr(_CTX, "slot", None)
    if slot is not None:
        t += f" slot {slot}"
    return f"[{t}] "


# ---- timestamped streams ------------------------------------------------------------------------------------------

class StampedStream:
    """A text stream whose lines start with the time and the writing thread's request tags.  A line is written whole
    when it ends (print() writes its text and its newline separately; other threads' lines never land in between)."""

    def __init__(self, out):
        self._out = out
        self._lock = threading.Lock()
        self._buf = threading.local()

    def write(self, s: str) -> int:
        buf = getattr(self._buf, "text", "") + s
        if "\n" in buf:
            *lines, buf = buf.split("\n")
            head = stamp() + " " + tags()
            with self._lock:
                self._out.write("".join(head + ln + "\n" for ln in lines))
        self._buf.text = buf
        return len(s)

    def flush(self) -> None:
        with self._lock:
            self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)


class StampedLog:
    """The engine's log file: every line with the time, to the millisecond (the engine's own lines come through a pipe,
    copy_lines; the server's lines into the same file through write)."""

    def __init__(self, f):
        self._f = f
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        lines = s.splitlines(keepends=True)
        head = stamp(ms=True) + " "
        with self._lock:
            for ln in lines:
                self._f.write(head + ln if ln.strip() else ln)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()

    @property
    def closed(self) -> bool:
        return self._f.closed


def copy_lines(stream, log: StampedLog) -> None:
    """A thread: the engine's stderr into its log, line by line.  Never stops reading while the engine runs (a full
    pipe would stop the engine); a closed log only drops the lines."""
    for line in iter(stream.readline, ""):
        try:
            log.write(line)
            log.flush()
        except (OSError, ValueError):
            pass


# ---- the conversation and the client ------------------------------------------------------------------------------

def conv_key(ids: list[int], assistant: list[int] | None) -> str | None:
    """The same short key for every turn of one conversation: the prompt up to its first assistant turn."""
    if not ids:
        return None
    end = min(len(ids), 2048)
    if assistant:
        n = len(assistant)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == assistant:
                end = i
                break
    return hashlib.sha1(struct.pack(f"<{end}i", *ids[:end])).hexdigest()[:6] if end else None


def trusted_nets(spec) -> list:
    """The proxies whose X-Forwarded-For is believed: addresses or networks (config "trusted_proxies")."""
    if not spec:
        return []
    items = spec if isinstance(spec, list) else str(spec).split(",")
    nets = []
    for s in items:
        s = str(s).strip()
        if s:
            nets.append(ipaddress.ip_network(s, strict=False))   # a bad entry stops the start (ValueError)
    return nets


def _trusted(addr: str, nets: list) -> bool:
    try:
        ip = ipaddress.ip_address(addr.strip().removeprefix("::ffff:"))
    except ValueError:
        return False
    return any(ip in n for n in nets)


def client_ip(peer: str, forwarded: str | None, nets: list) -> str:
    """The client's address: from the right of X-Forwarded-For, past every trusted proxy (an address a client wrote in
    the header itself is left of the first untrusted hop and never taken); the connection's own address otherwise."""
    if not nets or not forwarded or not _trusted(peer, nets):
        return peer
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for h in reversed(hops):
        if not _trusted(h, nets):
            return h
    return hops[0] if hops else peer


def client_kind(agent: str | None, referer: str | None) -> str:
    a = (agent or "").lower()
    if "opencode" in a:
        return "opencode"
    if "claude-cli" in a or "claude-code" in a:
        return "claude-code"
    if "mozilla" in a:       # through llama-swap: Strata's page is under /upstream/<model>/, llama-swap's own under /ui
        r = referer or ""
        return "strata-web" if "/upstream/" in r else "llama-swap-ui" if "/ui" in r else "browser"
    if a.startswith(("curl", "python", "wget", "go-http")):
        return a.split("/")[0]
    return a.split("/")[0][:24] if a else "unknown"


def key_print(auth: str | None) -> str | None:
    """Which API key, without the key: the first 6 hex of its SHA-256."""
    if not auth:
        return None
    key = auth.split(None, 1)[1] if " " in auth else auth
    return hashlib.sha256(key.encode()).hexdigest()[:6] if key else None


# ---- the event file -----------------------------------------------------------------------------------------------

class EventLog:
    """One JSON line per event in <folder>/<model>-YYYY-MM-DD.jsonl; files older than `days` go at the day's first."""

    def __init__(self, folder: str, model: str, days: int = 30):
        self.folder, self.days = folder, max(1, int(days))
        self.model = re.sub(r"[^A-Za-z0-9._-]+", "_", model or "model")
        self._lock = threading.Lock()
        self._day = None
        self._f = None
        os.makedirs(folder, exist_ok=True)

    def _file(self):
        day = time.strftime("%Y-%m-%d")
        if day != self._day:
            if self._f is not None:
                self._f.close()
            self._day = day
            self._f = open(os.path.join(self.folder, f"{self.model}-{day}.jsonl"), "a", encoding="utf-8")
            self._prune()
        return self._f

    def _prune(self) -> None:
        cutoff = time.time() - self.days * 86400
        for name in os.listdir(self.folder):
            if name.startswith(self.model + "-") and name.endswith(".jsonl"):
                p = os.path.join(self.folder, name)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                except OSError:
                    pass

    def write(self, event: str, **fields) -> None:
        rec = {"ts": stamp(ms=True), "ev": event, **{k: v for k, v in fields.items() if v is not None}}
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        try:
            with self._lock:
                f = self._file()
                f.write(line)
                f.flush()
        except OSError:
            pass
