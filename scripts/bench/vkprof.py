#!/usr/bin/env python3
# Sum the last N "Vulkan Timings" blocks of GGML_VK_PERF_LOGGER=1 output by op type
# (PERFORMANCE.md 7a). Usage: vkprof.py <log> [N=4]   (N=4: one llama-bench -p 2048 run)
import re, sys, collections
text = open(sys.argv[1], errors="replace").read()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
blocks = text.split("Vulkan Timings:")[1:][-n:]
tot = collections.Counter()
for b in blocks:
    for line in b.splitlines():
        m = re.match(r"(.+?): (\d+) x [\d.e+]+ us = ([\d.e+]+) us", line)
        if not m:
            continue
        name = m.group(1)
        key = name.split()[0] if not name.startswith(("MUL_MAT", "FLASH")) else name.split()[0]
        if name.startswith("MUL_MAT"):
            # tag quant type if present, e.g. "MUL_MAT_ID q4_K m=.. n=.."
            parts = name.split()
            key = " ".join(parts[:2]) if len(parts) > 1 and not parts[1][0].isdigit() and "=" not in parts[1] else parts[0]
        tot[key] += float(m.group(3))
s = sum(tot.values())
print(f"total {s/1000/len(blocks):.1f} ms per 512-token batch ({len(blocks)} batches)")
for k, v in tot.most_common(14):
    print(f"  {v/s*100:5.1f}%  {v/1000/len(blocks):7.1f} ms  {k}")
