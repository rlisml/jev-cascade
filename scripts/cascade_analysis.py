"""Confidence-cascade curves: kev-0.8b -> a larger kev model, plus a random-routing control.

Scans results/cascade/ for second-stage flat passes (`<source>_4b_pass1.json`,
`<source>_9b_pass1.json`) produced by cascade_kev_pass1.py and computes one set of
curves per stage. Cost units: kev-0.8b = 1, kev-4b = 5, kev-9b = 11.25 (parameter
ratios). Escalation = the lowest-0.8B-confidence fraction goes to the large model
(second pass = its flat pass-1).
Outputs: results/cascade/cascade_curves.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P_BAT = ROOT / "results" / "battery_kev"
OUT = ROOT / "results" / "cascade"
SOURCES = ["goemotions27", "massive60", "hwu64", "amzntoys60", "snips7"]
# stage tag -> (file suffix, cost-unit increment over the 0.8B forward)
STAGES = {"4b": ("_4b_pass1.json", 4.0), "9b": ("_9b_pass1.json", 10.25)}

out = {}
for stage, (suffix, unit) in STAGES.items():
    for src in SOURCES:
        f08 = P_BAT / f"{src}_pass1.json"
        fb = OUT / f"{src}{suffix}"
        if not (f08.exists() and fb.exists()):
            continue
        d08 = json.load(open(f08))["dists"]
        db = json.load(open(fb))["dists"]
        common = [rid for rid in d08 if rid in db]
        acc08 = {rid: float(max(d08[rid]["probs"], key=d08[rid]["probs"].get) == d08[rid]["label"])
                 for rid in common}
        accb = {rid: float(max(db[rid]["probs"], key=db[rid]["probs"].get) == db[rid]["label"])
                for rid in common}
        flat08 = float(np.mean(list(acc08.values())))
        flatb = float(np.mean(list(accb.values())))
        rows = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = int(round(frac * len(common)))
            esc = set(sorted(common, key=lambda kk: max(d08[kk]["probs"].values()))[:k])
            corr = [accb[rid] if rid in esc else acc08[rid] for rid in common]
            rows.append({"escalate_frac": frac, "acc": round(float(np.mean(corr)), 4),
                         "cost_units_08b": round(1.0 + frac * unit, 2),
                         "random_routing_acc": round((1 - frac) * flat08 + frac * flatb, 4)})
        out[f"{src}|{stage}"] = {
            "n": len(common), "flat_08b": round(flat08, 4), f"flat_{stage}": round(flatb, 4),
            "gap_pp": round(100 * (flatb - flat08), 1), "curve": rows}
        print(f"== {src} [{stage}] (0.8B {flat08:.4f} / {stage} {flatb:.4f}, "
              f"gap {100*(flatb-flat08):.1f}pp)")
        for r in rows:
            gate_gain = r["acc"] - r["random_routing_acc"]
            print(f"  r={r['escalate_frac']:.2f}: cascade={r['acc']:.4f} "
                  f"random={r['random_routing_acc']:.4f} gate_gain={gate_gain:+.4f} "
                  f"@{r['cost_units_08b']:.2f}u")

json.dump(out, open(OUT / "cascade_curves.json", "w"), indent=1)
print("saved cascade_curves.json")
