# Several conversations at once on a layer split (this fork, 4 x Tesla T4)

This fork's `main` runs up to 8 conversations at once on a 4-card layer split: agents working in the background
while a short chat question still gets its first token in a few seconds. Everything here is OFF unless switched on
in the run config's `env`; without these keys the engine behaves as upstream.

Tested on: 4 x Tesla T4 (sm_75, 16 GB, PCIe 3.0, 2 + 2 across two sockets), 2 x Xeon E5-2660 v2 (AVX only),
251 GiB RAM, Qwen3.8-Flash-Next Q8_0 and UD-Q4_K_XL, through llama-swap.

## The run config we use (Q8_0; Q4 is the same with its own split and pack)

Top level: `"parallel": 8`, `"layer_split": "10,22,35"` (Q4: `"11,24,36"`).

Args added to the usual ones: `--conversation-cache-mib 16384 --conversation-cache-slots 8
--conversation-cache-min-tokens 1024 --session-dir <dir> --kv int8 --kv-resident 32768 --spec 3`.

```json
"env": {
  "STRATA_STAGE_SERVE": "1",
  "STRATA_NUMA_PLACE": "1",
  "STRATA_PARALLEL_LOAD": "1",
  "STRATA_ELASTIC_SLOTS": "1",
  "STRATA_ELASTIC_SLOTS_GIVE": "1",
  "STRATA_ELASTIC_IDLE": "0",
  "STRATA_PIPED_MTP": "1",
  "STRATA_PIPED_MTP_DRAFTS": "2",
  "STRATA_SLOT_ADMIT": "1",
  "STRATA_SLOT_ADMIT_MULTI": "1",
  "STRATA_SLOT_ADMIT_CHUNK": "2048",
  "STRATA_P2P_HANDOFF": "1",
  "STRATA_SHORT_READS": "1",
  "STRATA_REPLY_REUSE": "1",
  "STRATA_READ_YIELD": "1",
  "STRATA_BATCH_READS": "1",
  "STRATA_BATCH_READS_MAX": "8",
  "STRATA_BATCH_READS_FAST": "512",
  "STRATA_BATCH_READS_FAST_SHRINK": "512",
  "STRATA_BATCH_READS_DECODE_WINDOWS": "0",
  "STRATA_FAST_LANE": "1",
  "STRATA_FAST_LANE_MAX": "96",
  "STRATA_NUMA_POOL": "0"
}
```

This box also needs `STRATA_NO_LARGEPAGES=1` (no large pages configured). `STRATA_NUMA_POOL=0` keeps upstream
0.2.2's automatic two-socket split off the engine's main pool: with STRATA_STAGE_SERVE every stage already has a pool on
its card's node, and the measurements below were taken that way.

Turing (sm_75) only, exact or within rounding: `STRATA_SM75_PERF=1 STRATA_SM75_STACK=1 STRATA_KQL=3
STRATA_SM75_Q8G=1 STRATA_GDN_KEYHEAD=1 STRATA_ARGMAX_ROWS_SM70=1 STRATA_BF16_TC=1
STRATA_BF16_TC_SKIP=indexer,router,ssm` (Q4 also `STRATA_MMQ_KQ=1`).

## Logging (server config keys)

`"trusted_proxies": ["127.0.0.1", "::1", "172.18.0.0/16"]` (the proxies in front: llama-swap, nginx's network) and
`"event_log": "/data/llm/logs/strata/events"`.  Every printed line and every engine-log line then starts with the time;
a request's lines carry `[req <id> conv <key> slot <n>]`; the event log has one JSON line per request (client address
and kind, prompt, reused, time to first token, the gap since the conversation's last request) and per conversation
parked or restored.  `"log_time": false` turns the times and tags off.  Known gap: `reused`/`source` in a request's
event can belong to another request when several overlap.

## What the switches do

| switch | what |
|---|---|
| STAGE_SERVE, PIPED_MTP | each card serves its layers on its own thread; slots in groups of 2 pipeline through the cards, MTP drafts while <= 2 conversations decode |
| ELASTIC_SLOTS(+GIVE) | a slot's K/V memory grows with its conversation, taken from the expert cache when the budget runs short |
| SLOT_ADMIT(+MULTI, CHUNK) | a new prompt is read beside the decoding slots, in 2048-token pieces |
| READ_YIELD | a read stops at a piece boundary for a text request waiting behind it |
| BATCH_READS | several reads share one prompt chunk |
| FAST_LANE | a prompt of <= 96 new tokens is read through the decode windows (first token in ~2-3 s under load) |
| slot_keep (automatic with PIPED_MTP) | an idle slot keeps its conversation; the server sends a continuation back to it |

## Measured (Q8_0, through llama-swap)

| | |
|---|---|
| one conversation decoding | ~29 tok/s |
| 1 / 2 / 4 / 8 conversations decoding | 24 / 32 / 43 / 63 tok/s total |
| 4 decoding beside a 30k-token read | ~30 tok/s total |
| short chat, first token, beside 5-6 coding agents | p50 2.6 s (was 17-24 s) |
| continuation of a held conversation (3k tokens) beside a decoder | 1.4 s |

Soak: 6 coding agents + a chat every 45 s, 22 min: 0 failures, 0 stalls, 0 engine errors.

## Known limits

- Fast-lane first-token logits differ from the prompt path's by more than chunk-size rounding (decode kernels vs
  prompt kernels); answers were coherent in every check.
- A browser over HTTP/1.1 opens at most 6 connections to one host: a reverse proxy in front should speak HTTP/2,
  or a 6th stream waits in the browser.
- The Monitor page's context-memory gauge needs `STRATA_ELASTIC_SLOTS` (the engine's KMEM lines).
