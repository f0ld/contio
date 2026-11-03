#!/usr/bin/env python3
"""M0 spikes: probe an OpenAI-compatible server (FastFlowLM, llama-server, ...)
for the behaviour contio depends on. Stdlib only. Prints one JSON object per test.

Usage: openai_probe.py <base_url> <model> [--embed-model TAG]
"""
import argparse
import json
import glob
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = (
    "The Framework Laptop 13 is a modular, repairable notebook. "
    "Its mainboard, ports and battery can be replaced by the owner. "
)


def find_ppt_sensor():
    """amdgpu hwmon 'PPT' = SoC package power (CPU+GPU+NPU), readable without root."""
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(f"{h}/name").read().strip() == "amdgpu":
                return f"{h}/power1_input"
        except OSError:
            pass
    return None


class PowerSampler:
    def __init__(self, path, interval=0.1):
        self.path, self.interval, self.samples = path, interval, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(int(open(self.path).read()) / 1e6)  # uW -> W
            except (OSError, ValueError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.path:
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    @property
    def avg_w(self):
        return sum(self.samples) / len(self.samples) if self.samples else None


PPT = find_ppt_sensor()


def post(base, path, body, stream=False, timeout=600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def emit(test, **kv):
    print(json.dumps({"test": test, **kv}), flush=True)


def bench_stream(base, model, prompt, max_tokens, label):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    usage = None
    try:
        with PowerSampler(PPT) as pw, post(base, "/v1/chat/completions", body, stream=True) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
    except urllib.error.HTTPError as e:
        emit(label, ok=False, error=f"HTTP {e.code}: {e.read()[:300]!r}")
        return
    t_end = time.perf_counter()
    gen = (usage or {}).get("completion_tokens", chunks)
    prompt_toks = (usage or {}).get("prompt_tokens")
    decode_s = (t_end - t_first) if t_first else None
    emit(
        label,
        ok=t_first is not None,
        prompt_tokens=prompt_toks,
        completion_tokens=gen,
        ttft_s=round(t_first - t0, 3) if t_first else None,
        prefill_tok_s=round(prompt_toks / (t_first - t0), 1) if t_first and prompt_toks else None,
        decode_tok_s=round((gen - 1) / decode_s, 2) if decode_s and gen > 1 else None,
        usage_reported=usage is not None,
        avg_soc_w=round(pw.avg_w, 2) if pw.avg_w else None,
        joules_per_token=round(pw.avg_w * (t_end - t0) / gen, 3) if pw.avg_w and gen else None,
    )


def test_tools(base, model):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in Karlsruhe right now?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "temperature": 0,
        # thinking models need room to reason before the call (M0: 512 was too small)
        "max_tokens": 2048,
    }
    try:
        with post(base, "/v1/chat/completions", body) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        emit("tool_call", ok=False, error=f"HTTP {e.code}: {e.read()[:300]!r}")
        return
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    args = None
    if calls:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, json.JSONDecodeError):
            args = "UNPARSEABLE"
    emit(
        "tool_call",
        ok=bool(calls) and isinstance(args, dict) and "city" in args,
        n_calls=len(calls),  # >1 identical calls = repetition loop
        completion_tokens=resp.get("usage", {}).get("completion_tokens"),
        finish_reason=resp["choices"][0].get("finish_reason"),
        tool_calls=calls,
        content_preview=(msg.get("content") or "")[:200],
    )


def test_json_schema(base, model):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Give me a person named Ada, age 36."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                },
            },
        },
        "temperature": 0,
        "max_tokens": 1024,
    }
    try:
        with post(base, "/v1/chat/completions", body) as r:
            content = json.load(r)["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        emit("json_schema", ok=isinstance(parsed.get("age"), int), content=content)
    except Exception as e:  # noqa: BLE001 - spike: record any failure
        emit("json_schema", ok=False, error=repr(e)[:300])


def test_embed_switch(base, embed_model):
    """Requesting a different model makes FLM swap. Measure the swap cost."""
    t0 = time.perf_counter()
    try:
        with post(base, "/v1/embeddings", {"model": embed_model, "input": ["hello npu"]}) as r:
            resp = json.load(r)
        if not resp.get("data"):
            emit("embed_switch", ok=False, error=f"no data in response: {json.dumps(resp)[:200]}")
            return
        dim = len(resp["data"][0]["embedding"])
        emit("embed_switch", ok=True, dim=dim, latency_s=round(time.perf_counter() - t0, 2))
    except Exception as e:  # noqa: BLE001
        emit("embed_switch", ok=False, error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--embed-model")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    try:
        emit("models", ok=True, models=[m["id"] for m in get(base, "/v1/models")["data"]])
    except Exception as e:  # noqa: BLE001
        emit("models", ok=False, error=repr(e)[:300])

    bench_stream(base, a.model, "Say hi.", 16, "warmup")
    bench_stream(base, a.model, "Write a 200-word story about a robot.", 256, "short_prompt")
    bench_stream(base, a.model, FILLER * 40 + "\nSummarize the text above in one sentence.", 64, "prompt_2k")
    bench_stream(base, a.model, FILLER * 160 + "\nSummarize the text above in one sentence.", 64, "prompt_8k")
    test_tools(base, a.model)
    test_json_schema(base, a.model)
    if a.embed_model:
        test_embed_switch(base, a.embed_model)
        # switch back so the next run starts from a known state
        bench_stream(base, a.model, "Say hi.", 16, "switch_back")


if __name__ == "__main__":
    sys.exit(main())
