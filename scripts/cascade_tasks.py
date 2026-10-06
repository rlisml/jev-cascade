"""Two-stage confidence cascade over HTTP-served decision models on hard tasks.

Protocol: the small model answers first over the full label menu; the
lowest-confidence fraction of records is escalated to the large model (full menu,
second pass). Because both passes are flat over the full menu, the second pass is
exactly the large model's flat pass — no extra queries beyond the two flat passes.
Tasks use 500 records per source (seed-0 sample of the test files) where both model
sizes are weak, so the cascade has headroom.

Stages (run with the corresponding model served at CASCADE_URL):
  flat : flat pass of the small model for both tasks
  flat2: flat pass of the large model for both tasks
  curve: offline, compute cascade curves
Usage: cascade_tasks.py <flat|flat2|curve>
Outputs: results/cascade_tasks/<task>_{small,large}_pass1.json, cascade_task_curves.json
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cascade_tasks"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
URL = os.environ.get("CASCADE_URL", "http://127.0.0.1:8796")
TASKS = ["goemotions27", "amzntoys60"]


def load_task(task):
    recs = [json.loads(l) for l in open(DATA / f"{task}_test.jsonl", encoding="utf-8")]
    idx = np.random.default_rng(0).choice(len(recs), size=500, replace=False)
    return [recs[i] for i in sorted(idx)]


def query(r, labels):
    payload = {"state": r["state"],
               "questions": {r["qid"]: {"type": "choice", "instructions": r["instr"],
                                        "criteria": {o: None for o in labels}}}}
    req = urllib.request.Request(f"{URL}/v1/systemone", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except Exception:
            if k == 2:
                raise
            time.sleep(1 + k)


def run_flat(suffix):
    for task in TASKS:
        f = OUT / f"{task}_{suffix}_pass1.json"
        if f.exists():
            print(f"[skip] {task}_{suffix}", flush=True)
            continue
        recs = load_task(task)
        labels = json.load(open(DATA / f"{task}_labels.json"))
        dist, t0 = {}, time.time()

        def work(r):
            o = query(r, labels)
            pr = o["answers"][r["qid"]]["probabilities"]
            return r["rid"], {"label": r["label"], "probs": pr}

        with ThreadPoolExecutor(max_workers=12) as ex:
            for rid, row in ex.map(work, recs):
                dist[rid] = row
                if len(dist) % 200 == 0:
                    print(f"{task}_{suffix}: {len(dist)}/500 ({time.time()-t0:.0f}s)",
                          flush=True)
        json.dump({"dists": dist, "labels": labels}, open(f, "w"))
        acc = float(np.mean([max(d["probs"], key=d["probs"].get) == d["label"]
                             for d in dist.values()]))
        print(f"[done] {task}_{suffix}: acc={acc:.4f}", flush=True)


def curve():
    out = {}
    for task in TASKS:
        fs = OUT / f"{task}_small_pass1.json"
        fl = OUT / f"{task}_large_pass1.json"
        if not (fs.exists() and fl.exists()):
            continue
        ds = json.load(open(fs))["dists"]
        dl = json.load(open(fl))["dists"]
        acc_s = {rid: float(max(d["probs"], key=d["probs"].get) == d["label"])
                 for rid, d in ds.items()}
        acc_l = {rid: float(max(d["probs"], key=d["probs"].get) == d["label"])
                 for rid, d in dl.items()}
        flat_s = float(np.mean(list(acc_s.values())))
        flat_l = float(np.mean(list(acc_l.values())))
        rows = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = int(round(frac * len(acc_s)))
            esc = set(sorted(acc_s, key=lambda kk: max(ds[kk]["probs"].values()))[:k])
            corr = [acc_l.get(rid, acc_s.get(rid)) if rid in esc else acc_s.get(rid)
                    for rid in acc_s]
            corr = [c for c in corr if c is not None]
            rows.append({"escalate_frac": frac, "acc": round(float(np.mean(corr)), 4),
                         "cost_units_small": round(1.0 + frac * 1.5, 3)})
        out[task] = {"flat_small": round(flat_s, 4), "flat_large": round(flat_l, 4),
                     "curve": rows}
        print(f"{task}: flat small={flat_s:.4f} flat large={flat_l:.4f}", flush=True)
        for r in rows:
            print(f"  r={r['escalate_frac']}: acc={r['acc']:.4f} "
                  f"@ {r['cost_units_small']:.2f}u", flush=True)
    json.dump(out, open(OUT / "cascade_task_curves.json", "w"), indent=1)
    print("saved cascade_task_curves.json", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "flat":
        run_flat("small")
    elif stage == "flat2":
        run_flat("large")
    else:
        curve()
