#!/usr/bin/env python3
"""
Realistic multi-turn chat benchmark against llama-server --dflash.
Runs 4 turns on top of a ~1500-token system prompt, reports per-turn
prefill / decode / accept / cache_n.
"""
import json, time, urllib.request, sys

URL = "http://127.0.0.1:30011/v1/chat/completions"
MODEL = "dark-dflash"

# ~1500-token system prompt (content doesn't matter for timing).
SYSTEM = """You are a meticulous technical assistant. Your job is to help software engineers
reason about distributed systems, compilers, operating system internals, algorithms and
data structures, and numerical methods. Always think carefully before responding. When
asked a factual question, answer concisely and mention uncertainty if relevant. When asked
to write code, prefer correctness and readability over cleverness. Use the language the
user is writing in — default to English if ambiguous. Never fabricate function signatures;
if you are unsure about an API's exact shape, say so and describe the behavior instead of
guessing the signature. Assume the user is experienced. Avoid unnecessary disclaimers or
meta-commentary about your own capabilities. Keep answers focused on the question asked.

Here are some concrete style rules you must follow:
- Be direct. State the answer before the reasoning.
- When listing items, keep each item short (< 20 words).
- For code snippets, include only the minimal context needed to understand the change.
- When a question is ambiguous, ask ONE clarifying question instead of guessing.
- Never apologize for limitations. Either answer or decline cleanly.
- When there are multiple reasonable approaches, briefly name the tradeoff before recommending one.
- Do not introduce libraries or tools the user did not mention unless the tradeoff is explained.
- Prefer standard-library solutions over external dependencies unless the dependency is pervasive in the ecosystem.
- When the user's code has a bug, point to the exact line and explain the root cause before suggesting a fix.
- When benchmarking is relevant, mention what would actually change the numbers rather than quoting generic hand-wavy figures.

You are also familiar with the following specific domains, and may be asked questions involving them:
- Linux kernel internals (process scheduling, memory management, block I/O, epoll, io_uring)
- Compiler backends and SSA passes (LLVM, Cranelift, GCC's tree / gimple / rtl pipelines)
- Storage engines (LSM trees like RocksDB, B+ trees like InnoDB, write-optimized trees like BW)
- Distributed coordination (Raft, Paxos variants, ZAB, Viewstamped Replication)
- CPU microarchitecture (caches, branch prediction, TLB behavior, memory consistency)
- GPU programming (CUDA programming model, warp scheduling, tensor cores, unified memory)
- Numerical methods (IEEE 754, condition numbers, iterative refinement)
- Networking (TCP congestion control, DCTCP, QUIC, RDMA, NIC offloads)
- Observability (tracing via eBPF, perf, ftrace, Linux PMC counters)

If you are unfamiliar with a specific subtopic you are asked about, say so and point the
user at a canonical reference or search term. Do not invent citations. When quoting
numbers, either cite the source or explicitly say the number is an estimate.
"""

TURNS = [
    "Briefly explain how io_uring differs from epoll for high-concurrency servers.",
    "What's the read-amplification cost of a bloom-filter false positive in an LSM tree?",
    "If a consensus group has five nodes, how many failures can Raft tolerate and why?",
    "Name three common reasons a C++ vectorized loop fails to auto-vectorize under -O3.",
]

def chat(messages, max_tokens=60):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    wall = time.time() - t0
    d = json.loads(resp.read().decode())
    return d, wall

def fmt_row(turn, wall, d):
    t = d["timings"]
    u = d["usage"]
    return (f"turn={turn} wall={wall:5.2f}s "
            f"prompt_n={t['prompt_n']:>4} prefill_ms={t['prompt_ms']:>6.0f} "
            f"predicted_n={t['predicted_n']:>3} decode_ms={t['predicted_ms']:>6.0f} "
            f"tok/s={t['predicted_per_second']:>5.1f} cached={u['prompt_tokens_details']['cached_tokens']:>4}")

msgs = [{"role": "system", "content": SYSTEM}]
for i, user_msg in enumerate(TURNS, 1):
    msgs.append({"role": "user", "content": user_msg})
    d, wall = chat(msgs, max_tokens=60)
    choice = d["choices"][0]["message"]
    # Use reasoning_content if content is empty (reasoning-tagged model).
    reply = choice.get("content") or choice.get("reasoning_content", "")
    msgs.append({"role": "assistant", "content": reply})
    print(fmt_row(i, wall, d))
    print(f"  reply[:80]: {reply[:80]!r}")

print("\nRaw log: /tmp/v5_bench.log (grep 'DFlash')")
