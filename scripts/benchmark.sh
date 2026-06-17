#!/bin/bash
# APExIA LLM Benchmark v3 — CONCURRENCY SWEEP
# Measures decode throughput at multiple concurrency levels against any OpenAI-compatible
# server (vLLM, etc.), reporting BOTH:
#   • per-user tok/s  — what a single user *feels* at that load (the UX number)
#   • aggregate tok/s — total system output (the capacity number)
# plus average request latency. N=1 is the single-stream figure.
#
# --prompt-tokens N pads a SHARED prefix to ~N tokens (identical across requests, so it's
# prefix-cacheable) with a unique tail per request — models a long system/schema prompt. Use it
# for the KV/context-bound concurrency curve; the default short prompt shows the compute-bound
# ceiling. The measured numbers for THIS model are in the model card.
#
# Usage:
#   ./benchmark.sh                                          # default sweep (n1, n16)
#   ./benchmark.sh --levels "1 16 32 64 128"               # custom levels
#   ./benchmark.sh --ceiling                               # wide sweep 1->128
#   ./benchmark.sh --prompt-tokens 6000 --levels "1 32 64 128"   # realistic long-prompt curve
#   ./benchmark.sh --url http://localhost:8011 --model qwen --max-tokens 256

URL="${LLAMA_URL:-http://localhost:8011}"
MODEL=""
MAX_TOKENS=256
PROMPT_TOKENS=0          # 0 = short prompt; >0 pads a shared (prefix-cacheable) prefix to ~N tokens
LEVELS="1 16"
FLOOR=20                # per-user tok/s floor for the "practical ceiling" call

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)           URL="$2";           shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --levels)        LEVELS="$2";        shift 2 ;;
        --floor)         FLOOR="$2";         shift 2 ;;
        --ceiling)       LEVELS="1 4 8 16 24 32 48 64 96 128"; shift ;;
        *) shift ;;
    esac
done

if [ -z "$MODEL" ]; then
    MODEL=$(curl -s --max-time 5 "$URL/v1/models" | python3 -c "import sys,json
try: print(json.load(sys.stdin)['data'][0]['id'])
except: print('')" 2>/dev/null)
    [ -z "$MODEL" ] && { echo "❌ Could not detect a model at $URL/v1/models — is the server up?"; exit 1; }
fi

echo "=================================================================="
echo "  APExIA LLM Benchmark v3 — concurrency sweep"
echo "  Server: $URL   Model: $MODEL   max_tokens: $MAX_TOKENS"
echo "  Levels: $LEVELS   |   floor: ${FLOOR} tok/s   |   prompt: ${PROMPT_TOKENS} tok (0=short)"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

URL="$URL" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" PROMPT_TOKENS="$PROMPT_TOKENS" \
LEVELS="$LEVELS" FLOOR="$FLOOR" python3 - <<'PY'
import os, json, time, urllib.request, concurrent.futures

URL = os.environ["URL"]; MODEL = os.environ["MODEL"]
MAXTOK = int(os.environ["MAX_TOKENS"]); FLOOR = float(os.environ["FLOOR"])
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "0"))
LEVELS = [int(x) for x in os.environ["LEVELS"].split()]

_BASE = "Write a long, detailed essay about the logistics of running a factory:"
# Shared prefix (~PROMPT_TOKENS tokens) — identical across requests so it's prefix-cacheable;
# each request appends a unique tail. Models a long system/schema prompt.
_FILLER = ("Context block used only to pad the shared prefix to the target length so the "
           "prefix-cache and KV behavior match a long real-world system prompt. ")
PREFIX = ((_FILLER * (PROMPT_TOKENS * 4 // len(_FILLER) + 1))[:PROMPT_TOKENS * 4]
          if PROMPT_TOKENS > 0 else "")

def one(uid):
    prompt = (PREFIX + f"\n[request {uid}] " + _BASE) if PROMPT_TOKENS > 0 else _BASE
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAXTOK,
                       "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", MAXTOK)
    return ct, dt

print(f"  warming up... (prompt ~{PROMPT_TOKENS or 30} tok)"); one(0); one(1)
print()
print(f"  {'users':>5} | {'per-user tok/s':>14} | {'aggregate tok/s':>15} | {'avg latency':>11}")
print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*15}-+-{'-'*11}")

practical_max = LEVELS[0]
for n in LEVELS:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        w0 = time.time()
        res = list(ex.map(one, range(n)))
        wall = time.time() - w0
    total = sum(r[0] for r in res)
    per_user = sum(r[0] / r[1] for r in res) / len(res)
    agg = total / wall
    lat = sum(r[1] for r in res) / len(res)
    flag = "  ← below usable floor" if per_user < FLOOR else ""
    if per_user >= FLOOR:
        practical_max = n
    print(f"  {n:>5} | {per_user:>14.1f} | {agg:>15.0f} | {lat:>9.2f}s{flag}")

print()
print(f"  ➤ Practical ceiling (per-user stays ≥ {FLOOR:.0f} tok/s): ~{practical_max} concurrent users")
PY
echo "=================================================================="
