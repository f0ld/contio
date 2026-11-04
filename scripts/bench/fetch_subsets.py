#!/usr/bin/env python3
"""Fetch the fixed quality-benchmark subsets into registry/bench/ (spec §8.2).

Small, seeded samples so runs are quick and comparable across machines:
  reasoning_gsm8k.json      50 GSM8K test problems           (openai/gsm8k, MIT)
  knowledge_mmlu_pro.json   4 per subject, 56 MMLU-Pro items  (TIGER-Lab/MMLU-Pro, MIT)
  coding_humaneval.json     40 HumanEval tasks                (openai/openai_humaneval, MIT)

Stdlib only. Re-running produces the same files.
    python3 scripts/bench/fetch_subsets.py
"""
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "registry" / "bench"
API = "https://datasets-server.huggingface.co"
SEED = 20261104


def get(endpoint, **params):
    q = urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{API}/{endpoint}?{q}", timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - rate limits and transient Hub errors; back off
            print(f"  retry {endpoint} {params.get('offset')}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"can't fetch {endpoint} {params}")


def all_rows(dataset, config, split, total):
    out = []
    for off in range(0, total, 100):
        out += [x["row"] for x in get("rows", dataset=dataset, config=config, split=split, offset=off, length=100)["rows"]]
        time.sleep(0.3)
    return out


def write(name, source, license_, items):
    doc = {"source": source, "license": license_, "seed": SEED, "items": items}
    (OUT / name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{name}: {len(items)} items")


def gsm8k():
    data = all_rows("openai/gsm8k", "main", "test", 1319)
    pick = random.Random(SEED).sample(range(len(data)), 50)
    items = []
    for i in sorted(pick):
        answer = data[i]["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k-{i}", "question": data[i]["question"], "answer": float(answer)})
    write("reasoning_gsm8k.json", "https://huggingface.co/datasets/openai/gsm8k (test split)", "MIT", items)


def mmlu_pro():
    data = all_rows("TIGER-Lab/MMLU-Pro", "default", "test", 12032)
    by_cat = {}
    for i, r in enumerate(data):
        by_cat.setdefault(r["category"], []).append(i)
    rng = random.Random(SEED)
    items = []
    for cat in sorted(by_cat):
        for i in sorted(rng.sample(by_cat[cat], 4)):
            r = data[i]
            items.append({
                "id": f"mmlu-pro-{r['question_id']}",
                "category": cat,
                "question": r["question"],
                "options": r["options"],
                "answer": r["answer"],
            })
    write("knowledge_mmlu_pro.json", "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro (test split)", "MIT", items)


def humaneval():
    data = all_rows("openai/openai_humaneval", "openai_humaneval", "test", 164)
    pick = random.Random(SEED).sample(range(len(data)), 40)
    items = []
    for i in sorted(pick):
        r = data[i]
        items.append({
            "id": r["task_id"].replace("/", "-"),
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        })
    write("coding_humaneval.json", "https://huggingface.co/datasets/openai/openai_humaneval", "MIT", items)


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or ["gsm8k", "mmlu_pro", "humaneval"]
    for name in wanted:
        globals()[name]()
