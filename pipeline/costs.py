"""Summarise runs.jsonl into the measured tables for COSTS.md.

    make costs        # prints markdown; paste or redirect into COSTS.md
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

from .common import RUNS


def main() -> None:
    if not RUNS.exists():
        print("runs.jsonl is empty")
        return
    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["stage"], r["model"])].append(r)
    print("| stage | model | calls | ok | errors | median latency s | p90 latency s | GPU min | list-price USD |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    tot_gpu = tot_cost = 0.0
    for (stage, model), rs in sorted(by.items()):
        lat = sorted(r["latency_ms"] / 1000 for r in rs if r.get("latency_ms") and r["status"] == "ok")
        gpu = sum(r.get("gpu_s") or 0 for r in rs) / 60
        cost = sum(r.get("cost_usd_list_price") or 0 for r in rs)
        tot_gpu += gpu
        tot_cost += cost
        ok = sum(r["status"] == "ok" for r in rs)
        med = f"{statistics.median(lat):.1f}" if lat else "-"
        p90 = f"{lat[min(len(lat) - 1, int(len(lat) * 0.9))]:.1f}" if lat else "-"
        print(f"| {stage} | `{model}` | {len(rs)} | {ok} | {len(rs) - ok} | {med} | {p90} | {gpu:.1f} | {cost:.4f} |")
    print(f"| **total** | | {len(rows)} | | | | | **{tot_gpu:.1f}** | **{tot_cost:.4f}** |")
    runs = defaultdict(int)
    for r in rows:
        if r["provider"] != "local":
            runs[r["run_id"]] += 1
    print("\nModel calls per pipeline run (a re-run with no storyboard change should be 0):\n")
    for rid, n in runs.items():
        print(f"- `{rid}`: {n}")


if __name__ == "__main__":
    main()
