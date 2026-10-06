"""Probe battery of candidate-menu operations on Open-Jev-2B (vllm-jev HTTP endpoint).

Same cells as battery_kev.py (flat / random15 / random15_free / avoid15 / avoid15_free /
self15{clean,verify} / self2_clean); serves as the cross-family spot check.

Usage: battery_jev.py <source>   (vllm-jev endpoint via BATTERY_URL, default localhost)
Outputs: results/battery_jev/jev2b_<source>_<cell>.json (+ jev2b_<source>_pass1.json)
"""
import json
import os
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "battery_jev"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
URL = os.environ.get("BATTERY_URL", "http://127.0.0.1:8796")
VERIFY_INSTR = "Which of these {C} options matches best?"


def rng_for(key):
    return np.random.default_rng(zlib.crc32(key.encode()))


def query(payload, retries=3):
    req = urllib.request.Request(f"{URL}/v1/systemone",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1 + k)


def argmax_key(dist):
    return max(dist, key=dist.get)


def run_cell(model_key, tag, items, save_full=False, workers=12):
    f = OUT / f"{tag}.json"
    if f.exists():
        print(f"[skip] {tag}", flush=True)
        return json.load(open(f))["rows"]

    def work(i):
        it = items[i]
        payload = {"state": it["state"],
                   "questions": {it["qid"]: {"type": "choice", "instructions": it["instr"],
                                             "criteria": {o: None for o in it["menu"]}}}}
        o = query(payload)
        pr = o["answers"][it["qid"]]["probabilities"]
        row = {"rid": it["rid"], "correct": float(argmax_key(pr) == it["label"]),
               "conf": round(max(pr.values()), 5),
               "p2_gold": round(pr.get(it["label"], 0.0), 5),
               "gold_in_menu": it["label"] in it["menu"],
               "top10": {k: round(v, 5) for k, v in sorted(pr.items(), key=lambda kv: -kv[1])[:10]},
               "menu": it["menu"]}
        if save_full:
            row["full"] = {k: round(v, 6) for k, v in pr.items()}
        return i, row

    rows, t0 = [None] * len(items), time.time()
    # gold-free cells: out-of-menu records are correct=0 by construction, skip the call
    for j, it in enumerate(items):
        if it["label"] not in it["menu"]:
            rows[j] = {"rid": it["rid"], "correct": 0.0, "conf": 0.0, "p2_gold": 0.0,
                       "gold_in_menu": False, "top10": {}, "menu": it["menu"]}
    live = [i for i in range(len(items)) if rows[i] is None]
    done_n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, row in ex.map(work, live):
            rows[i] = row
            done_n += 1
            if done_n % 250 == 0:
                print(f"{tag}: {done_n}/{len(live)} ({time.time()-t0:.0f}s)", flush=True)
    rows = [r for r in rows if r]
    acc = float(np.mean([r["correct"] for r in rows])) if rows else float("nan")
    json.dump({"tag": tag, "n": len(rows), "acc": round(acc, 4), "rows": rows}, open(f, "w"))
    print(f"[done] {tag}: acc={acc:.4f} n={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


source = sys.argv[1]
recs = [json.loads(l) for l in open(DATA / f"{source}_test.jsonl", encoding="utf-8")]
labels = json.load(open(DATA / f"{source}_labels.json"))
print(f"== jev2b {source}: {len(recs)} records, {len(labels)} labels ==", flush=True)
model_key = "jev2b"

# ---- flat pass-1 over the full label menu ----
f1 = OUT / f"{model_key}_{source}_pass1.json"
if f1.exists():
    dist = json.load(open(f1))["dists"]
else:
    items = [{"rid": r["rid"], "state": r["state"], "qid": r["qid"], "instr": r["instr"],
              "label": r["label"], "menu": labels} for r in recs]
    rows = run_cell(model_key, f"{model_key}_{source}_flat", items, save_full=True)
    dist = {r["rid"]: {"label": next(it["label"] for it in items if it["rid"] == r["rid"]),
                       "probs": r["full"]} for r in rows}
    json.dump({"dists": dist}, open(f1, "w"))
    print(f"[done] {model_key}_{source}_pass1: n={len(dist)}", flush=True)

have = [r for r in recs if r["rid"] in dist]
print(f"cells over {len(have)}/{len(recs)} records (pass-1 coverage)", flush=True)


def self_menu(C, gold, probs, rid, tag):
    rivals = [o for o in sorted(probs, key=probs.get, reverse=True) if o != gold][: C - 1]
    menu = rivals + [gold]
    rng_for(f"{tag}|shuf|{rid}").shuffle(menu)
    return menu


for tag, kind in (("random15", "random"), ("random15_free", "random_free"),
                  ("avoid15", "avoid"), ("avoid15_free", "avoid_free"),
                  ("self15_clean", "self"), ("self15_verify", "self"),
                  ("self2_clean", "self2")):
    C = min(15, len(labels)) if "15" in tag else 2
    items = []
    for r in have:
        g, pr, rid = r["label"], dist[r["rid"]]["probs"], r["rid"]
        if kind == "random":
            rivals = list(rng_for(f"{source}|{tag}|{rid}").choice(
                [o for o in labels if o != g], size=C - 1, replace=False))
            menu = rivals + [g]
            rng_for(f"{source}|{tag}|shuf|{rid}").shuffle(menu)
        elif kind == "random_free":
            menu = list(rng_for(f"{source}|{tag}|{rid}").choice(labels, size=C, replace=False))
        elif kind == "avoid":
            pool_ = [o for o in labels if o != g]
            w = np.array([1.0 / (pr.get(o, 0.0) + 0.02) for o in pool_])
            w = w / w.sum()
            picks = list(rng_for(f"{source}|{tag}|{rid}").choice(pool_, size=C - 1,
                                                                 replace=False, p=w))
            menu = picks + [g]
            rng_for(f"{source}|{tag}|shuf|{rid}").shuffle(menu)
        elif kind == "avoid_free":
            w = np.array([1.0 / (pr.get(o, 0.0) + 0.02) for o in labels])
            w = w / w.sum()
            menu = list(rng_for(f"{source}|{tag}|{rid}").choice(labels, size=C,
                                                                replace=False, p=w))
        else:
            menu = self_menu(C, g, pr, rid, f"{model_key}_{source}_{tag}")
        instr = VERIFY_INSTR.format(C=C) if tag.endswith("verify") else r["instr"]
        items.append({"rid": rid, "state": r["state"], "qid": r["qid"], "instr": instr,
                      "label": g, "menu": list(menu)})
    if items:
        run_cell(model_key, f"{model_key}_{source}_{tag}", items)

print(f"battery_jev {source} done", flush=True)
