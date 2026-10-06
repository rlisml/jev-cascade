"""Confidence cascade second pass: kev-0.8b -> a larger kev model, flat pass over one source.

The 0.8B pass-1 distributions come from battery_kev.py (results/battery_kev/<source>_pass1.json).
This script runs the missing large-model flat pass-1 per source; cascade curves are
then computed offline by cascade_analysis.py (second pass = large flat = its pass-1).
The second-stage model defaults to jaredpalmer/kev-4b; override with MODEL_STAGE2
(e.g. jaredpalmer/kev-9b). Cost units in cascade_analysis.py assume the 4B default.

Usage: cascade_kev_pass1.py <source>   (GPU via CUDA_VISIBLE_DEVICES)
Requires: the kev package on PYTHONPATH (set KEV_ROOT to the repo checkout).
Outputs: results/cascade/<source>_<stage>_pass1.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cascade"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_kev_root = os.environ.get("KEV_ROOT")
if _kev_root:
    sys.path.insert(0, _kev_root)

from kev.model import ContextOverflow
from kev.predictors import LocalPredictor

source = sys.argv[1]
STAGE2 = os.environ.get("MODEL_STAGE2", "jaredpalmer/kev-4b")
stage = {"jaredpalmer/kev-4b": "4b", "jaredpalmer/kev-9b": "9b"}.get(STAGE2, "stage2")
f = OUT / f"{source}_{stage}_pass1.json"
if f.exists():
    print(f"[skip] {source}", flush=True)
    sys.exit(0)

recs = [json.loads(l) for l in open(DATA / f"{source}_test.jsonl", encoding="utf-8")]
labels = json.load(open(DATA / f"{source}_labels.json"))
print(f"== {source}: {len(recs)} records, {len(labels)} labels, {STAGE2} ==", flush=True)

p = LocalPredictor(STAGE2, "cuda")
dist, n_ovf, t0 = {}, 0, time.time()
for i, r in enumerate(recs):
    rec = {"state": r["state"],
           "questions": {r["qid"]: {"type": "choice", "instructions": r["instr"],
                                    "criteria": {o: None for o in labels},
                                    "label": r["label"], "src": f"cascade_{source}"}},
           "_meta": {"source": source, "variant": "clean", "id": r["rid"],
                     "group_id": r["rid"], "row": r["rid"]}}
    try:
        o = p(rec)
    except ContextOverflow:
        n_ovf += 1
        continue
    dist[r["rid"]] = {"label": r["label"], "probs": o["probabilities"][r["qid"]]}
    if (i + 1) % 500 == 0:
        print(f"{source}_{stage}: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
acc = float(np.mean([max(d["probs"], key=d["probs"].get) == d["label"]
                     for d in dist.values()]))
json.dump({"dists": dist, "labels": labels, "acc": round(acc, 4)}, open(f, "w"))
print(f"[done] {source}_{stage}: acc={acc:.4f} n={len(dist)} ovf={n_ovf} "
      f"({time.time()-t0:.0f}s)", flush=True)
