#!/usr/bin/env python3
"""Regenerate registry/geometry.json from the GGUF headers of the curated repos.

For every model in registry/catalog.toml this reads the metadata of one GGUF in
the repo with HTTP Range requests (it stops before the tokenizer arrays, so a
few hundred KiB per model) and records what the fit estimator needs:
architecture, max context, parameter count, KV cache bytes per token, and
whether the chat template supports tools / reasoning.

    python3 scripts/catalog/geometry.py            # all models
    python3 scripts/catalog/geometry.py qwen3-8b   # only these ids

Set HF_TOKEN for gated repos.
"""
import json
import os
import re
import struct
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "registry" / "catalog.toml"
OUT = ROOT / "registry" / "geometry.json"
HF = "https://huggingface.co"
F16 = 2  # bytes per KV element (llama.cpp default cache type)
UBATCH = 512


def get(url, headers=None):
    h = {"User-Agent": "contio-geometry"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h)) as r:
        return r.read()


class RangeReader:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0

    def read(self, n):
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            self.buf += get(self.url, {"Range": f"bytes={start}-{start + (256 << 10) - 1}"})
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def value(r, t):
    if t in SCALAR:
        return r.u(SCALAR[t])
    if t == 8:
        return r.read(r.u("Q")).decode("utf-8", "replace")
    if t == 9:
        et, n = r.u("I"), r.u("Q")
        return [value(r, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF type {t}")


def header(url):
    """Metadata up to the first tokenizer key (plus the chat template)."""
    r = RangeReader(url)
    if r.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.u("I"), r.u("Q")
    meta = {}
    for _ in range(r.u("Q")):
        key = r.read(r.u("Q")).decode()
        if key.startswith("tokenizer.") and key != "tokenizer.chat_template":
            if "general.architecture" in meta and any(k.endswith(".block_count") for k in meta):
                # template usually comes last; tokens are megabytes, so stop here
                break
        meta[key] = value(r, r.u("I"))
    return meta, len(r.buf)


def per_layer(v, n):
    return list(v) if isinstance(v, list) else [v] * n


def kv_geometry(m):
    """KV cache bytes per token: full-attention layers and sliding-window layers."""
    a = m["general.architecture"]
    g = lambda k, d=None: m.get(f"{a}.{k}", d)
    n = g("block_count")
    heads = g("attention.head_count")
    emb = g("embedding_length")
    kv_heads = per_layer(g("attention.head_count_kv", heads), n)
    k_len = g("attention.key_length", emb // heads if heads else 0)
    v_len = g("attention.value_length", k_len)
    k_swa = g("attention.key_length_swa", k_len)
    v_swa = g("attention.value_length_swa", v_len)
    window = g("attention.sliding_window")

    pattern = g("attention.sliding_window_pattern")
    if isinstance(pattern, list):
        swa = [bool(p) for p in pattern]
    elif isinstance(pattern, int) and pattern > 0:
        swa = [(i + 1) % pattern != 0 for i in range(n)]
    elif a == "gpt-oss" and window:
        swa = [i % 2 == 0 for i in range(n)]  # llama.cpp: even layers use the window
    else:
        swa = [bool(window)] * n if a in ("gemma2",) else [False] * n

    interval = g("full_attention_interval")  # hybrid linear-attention models
    has_kv = [(i + 1) % interval == 0 for i in range(n)] if interval else [True] * n
    shared = g("attention.shared_kv_layers", 0) or 0  # Gemma 3n/4 E-models reuse KV
    for i in range(n - shared, n):
        has_kv[i] = False

    lora = g("attention.kv_lora_rank")  # MLA (DeepSeek2 / GLM-4.7-Flash): cache latents
    rope = g("rope.dimension_count", 0)

    full = swa_b = 0
    for i in range(n):
        if not has_kv[i] or kv_heads[i] == 0:
            continue
        if lora:
            b = (lora + rope) * F16
        elif swa[i]:
            b = kv_heads[i] * (k_swa + v_swa) * F16
        else:
            b = kv_heads[i] * (k_len + v_len) * F16
        if swa[i] and window:
            swa_b += b
        else:
            full += b
    return {"full_bytes_per_token": full, "swa_bytes_per_token": swa_b, "swa_window": (window or 0)}


AUX_PREFIXES = ("mtp-", "eagle", "dflash", "mmproj", "imatrix", "draft")
QUANT_START = re.compile(r"^(i?q\d|tq\d|mxfp4|bf16|f16|f32|fp16|fp8)", re.I)


def is_aux(path):
    name = path.rsplit("/", 1)[-1].lower()
    return name.startswith(AUX_PREFIXES) or "/mtp/" in "/" + path.lower()


def quant_of(path):
    """Same rule as contio-catalog: last quant-looking token of the file stem."""
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", path.rsplit("/", 1)[-1][: -len(".gguf")])
    tokens = re.split(r"[-.]", stem)
    for i in range(len(tokens) - 1, -1, -1):
        if QUANT_START.match(tokens[i]):
            q = tokens[i].upper()
            return f"UD-{q}" if i > 0 and tokens[i - 1].upper() == "UD" else q
    return None


def probe_file(repo, recommended):
    """The recommended quant's (first) file, else the smallest model file."""
    tree = json.loads(get(f"{HF}/api/models/{repo}/tree/main?recursive=true"))
    ggufs = [f for f in tree if f["type"] == "file" and f["path"].endswith(".gguf") and not is_aux(f["path"])]
    if not ggufs:
        raise ValueError("no model GGUF in repo")
    first = [f for f in ggufs if "-of-" not in f["path"] or "-00001-of-" in f["path"]]
    rec = [f for f in first if quant_of(f["path"]) == (recommended or "").upper()]
    return (rec or sorted(first, key=lambda f: f["size"]))[0]["path"]


def main(ids):
    cat = tomllib.loads(CATALOG.read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    total = 0
    for model in cat["models"]:
        if ids and model["id"] not in ids:
            continue
        if "ggml" in model:  # whisper.cpp: no GGUF header, no KV cache
            continue
        repo = model["repo"]
        try:
            info = json.loads(get(f"{HF}/api/models/{repo}?expand[]=gguf"))
            path = probe_file(repo, model.get("recommended"))
            meta, read = header(f"{HF}/{repo}/resolve/main/{path}")
            if "general.architecture" not in meta:
                raise ValueError(f"{path} has no architecture key")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"!! {model['id']:32} {repo}: {e}", file=sys.stderr)
            continue
        total += read
        arch = meta["general.architecture"]
        tmpl = meta.get("tokenizer.chat_template") or (info.get("gguf") or {}).get("chat_template") or ""
        out[repo] = {
            "arch": arch,
            "context_length": meta.get(f"{arch}.context_length"),
            "params": (info.get("gguf") or {}).get("total"),
            "kv": kv_geometry(meta),
            "tools": "tools" in tmpl,
            "reasoning": "think" in tmpl or "reasoning" in tmpl,
        }
        out[repo]["probed"] = path
        kv = out[repo]["kv"]
        if kv["full_bytes_per_token"] + kv["swa_bytes_per_token"] == 0:
            print(f"!! {model['id']}: no KV cache found in {path}", file=sys.stderr)
        print(f"{model['id']:32} {arch:16} ctx={out[repo]['context_length']!s:8} "
              f"kv/tok={kv['full_bytes_per_token'] / 1024:7.1f} KiB swa={kv['swa_bytes_per_token'] / 1024:6.1f} KiB"
              f"@{kv['swa_window']:<5} ({read >> 10} KiB read)")
    OUT.write_text(json.dumps(dict(sorted(out.items())), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({total >> 20} MiB of headers read)")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
