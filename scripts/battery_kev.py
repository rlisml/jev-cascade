"""Probe battery of candidate-menu operations on a kev decision model (local predictor).

Per source: a flat pass over the full label menu (full distributions saved), then the
menu cells
  random15        : uniform rivals + gold, shuffled            (restriction-baseline cell)
  random15_free   : 15 uniform labels, gold NOT forced         (gold-free deployment cell)
  avoid15         : confusion-avoidant rivals + gold, shuffled (weights ~ 1/(p1+0.02))
  avoid15_free    : confusion-avoidant, gold NOT forced        (gold-free deployment cell)
  self15_clean    : pass-1 top rivals + gold, task instruction
  self15_verify   : same menus, comparison-style wording       (presentation-effect cell)
  self2_clean     : top-1 rival + gold
Residuals against the restriction prediction are computed by battery_analysis.py.

Usage: battery_kev.py <source>   (GPU via CUDA_VISIBLE_DEVICES)
Requires: the kev package on PYTHONPATH (set KEV_ROOT to the repo checkout).
Outputs: results/battery_kev/<source>_<cell>.json (+ <source>_pass1.json)
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "battery_kev"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
MODEL = "jaredpalmer/kev-0.8b"
VERIFY_INSTR = "Which of these {C} options matches best?"

_kev_root = os.environ.get("KEV_ROOT")
if _kev_root:
    sys.path.insert(0, _kev_root)

from kev.model import ContextOverflow
from kev.predictors import LocalPredictor


def rng_for(key):
    return np.random.default_rng(zlib.crc32(key.encode()))


def load_source(source):
    with open(DATA / f"{source}_test.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_cell(source, tag, rows):
    acc = float(np.mean([r["correct"] for r in rows])) if rows else float("nan")
    with open(OUT / f"{source}_{tag}.json", "w") as f:
        json.dump({"tag": tag, "n": len(rows), "acc": round(acc, 4), "rows": rows}, f)
    print(f"[done] {source}_{tag}: acc={acc:.4f} n={len(rows)}", flush=True)


def run_cell(model, source, tag, items):
    f = OUT / f"{source}_{tag}.json"
    if f.exists():
        print(f"[skip] {source}_{tag}", flush=True)
        return
    rows, n_ovf, t0 = [], 0, time.time()
    # gold-free cells: records whose menu misses gold are correct=0 by construction —
    # no model call needed (deployment semantics); the local predictor requires the
    # label to be in the menu, so only in-menu records go through it
    live = [it for it in items if it["label"] in it["menu"]]
    for i, it in enumerate(live):
        rec = {"state": it["state"],
               "questions": {it["qid"]: {"type": "choice", "instructions": it["instr"],
                                         "criteria": {o: None for o in it["menu"]},
                                         "label": it["label"], "src": f"{source}_{tag}"}},
               "_meta": {"source": source, "variant": "clean", "id": it["rid"],
                         "group_id": it["rid"], "row": it["rid"]}}
        try:
            o = model(rec)
        except ContextOverflow:
            n_ovf += 1
            continue
        pr = o["probabilities"][it["qid"]]
        rows.append({"rid": it["rid"], "correct": float(argmax_key(pr) == it["label"]),
                     "conf": round(max(pr.values()), 5),
                     "p2_gold": round(pr.get(it["label"], 0.0), 5),
                     "gold_in_menu": it["label"] in it["menu"], "C": len(it["menu"]),
                     "top10": {k: round(v, 5) for k, v in
                               sorted(pr.items(), key=lambda kv: -kv[1])[:10]},
                     "menu": it["menu"]})
        if (i + 1) % 1000 == 0:
            print(f"{source}_{tag}: {i+1}/{len(live)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"{source}_{tag}: done ({time.time()-t0:.0f}s, ovf={n_ovf})", flush=True)
    live_ids = {it["rid"] for it in live}
    for it in items:
        if it["rid"] not in live_ids:
            rows.append({"rid": it["rid"], "correct": 0.0, "conf": 0.0, "p2_gold": 0.0,
                         "gold_in_menu": False, "C": len(it["menu"]),
                         "top10": {}, "menu": it["menu"]})
    save_cell(source, tag, rows)


def argmax_key(dist):
    return max(dist, key=dist.get)


source = sys.argv[1]
recs = load_source(source)
labels = json.load(open(DATA / f"{source}_labels.json"))
print(f"== {source}: {len(recs)} records, {len(labels)} labels, model={MODEL} ==", flush=True)

p = LocalPredictor(MODEL, "cuda")

# ---- flat pass-1 over the full label menu ----
f1 = OUT / f"{source}_pass1.json"
if f1.exists():
    dist = json.load(open(f1))["dists"]
else:
    dist, t0 = {}, time.time()
    for i, r in enumerate(recs):
        rec = {"state": r["state"],
               "questions": {r["qid"]: {"type": "choice", "instructions": r["instr"],
                                        "criteria": {o: None for o in labels},
                                        "label": r["label"], "src": f"{source}_flat"}},
               "_meta": {"source": source, "variant": "clean", "id": r["rid"],
                         "group_id": r["rid"], "row": r["rid"]}}
        try:
            o = p(rec)
        except ContextOverflow:
            continue
        dist[r["rid"]] = {"label": r["label"], "probs": o["probabilities"][r["qid"]]}
        if (i + 1) % 500 == 0:
            print(f"{source}_flat: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"dists": dist}, open(f1, "w"))
    acc = np.mean([argmax_key(d["probs"]) == d["label"] for d in dist.values()])
    print(f"[done] {source}_flat: acc={acc:.4f} n={len(dist)}", flush=True)

# menu cells run over the records covered by pass-1
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
    # narrow label spaces: cap C at the label-set size (recorded per row; gates compare
    # at matched C)
    C15 = min(15, len(labels))
    items = []
    for r in have:
        g, pr, rid = r["label"], dist[r["rid"]]["probs"], r["rid"]
        C = C15 if "15" in tag else 2
        if kind == "random":
            rivals = list(rng_for(f"{source}|{tag}|{rid}").choice(
                [o for o in labels if o != g], size=C - 1, replace=False))
            menu = rivals + [g]
            rng_for(f"{source}|{tag}|shuf|{rid}").shuffle(menu)
        elif kind == "random_free":
            menu = list(rng_for(f"{source}|{tag}|{rid}").choice(labels, size=C, replace=False))
        elif kind == "avoid":
            pool = [o for o in labels if o != g]
            w = np.array([1.0 / (pr.get(o, 0.0) + 0.02) for o in pool])
            w = w / w.sum()
            picks = list(rng_for(f"{source}|{tag}|{rid}").choice(pool, size=C - 1,
                                                                 replace=False, p=w))
            menu = picks + [g]
            rng_for(f"{source}|{tag}|shuf|{rid}").shuffle(menu)
        elif kind == "avoid_free":
            w = np.array([1.0 / (pr.get(o, 0.0) + 0.02) for o in labels])
            w = w / w.sum()
            menu = list(rng_for(f"{source}|{tag}|{rid}").choice(labels, size=C,
                                                                replace=False, p=w))
        else:  # self / self2
            menu = self_menu(C, g, pr, rid, f"{source}_{tag}")
        instr = VERIFY_INSTR.format(C=C) if tag.endswith("verify") else r["instr"]
        items.append({"rid": rid, "state": r["state"], "qid": r["qid"], "instr": instr,
                      "label": g, "menu": list(menu)})
    if items:
        run_cell(p, source, tag, items)

print(f"battery_kev {source} done", flush=True)
