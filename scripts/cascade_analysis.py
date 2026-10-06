"""Confidence-cascade curves: kev-0.8b -> kev-4b across sources, plus a random-routing control.

Cost units: kev-0.8b = 1, kev-4b = 5 (parameter ratio). Escalation = the lowest-0.8B-
confidence fraction goes to the 4B model (second pass = its flat pass-1).
Outputs: results/cascade/cascade_curves.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P_BAT = ROOT / "results" / "battery_kev"
OUT = ROOT / "results" / "cascade"
SOURCES = ["goemotions27", "massive60", "hwu64", "amzntoys60", "snips7"]

out = {}
for src in SOURCES:
    f08 = P_BAT / f"{src}_pass1.json"
    f4b = OUT / f"{src}_4b_pass1.json"
    if not (f08.exists() and f4b.exists()):
        continue
    d08 = json.load(open(f08))["dists"]
    d4b = json.load(open(f4b))["dists"]
    common = [rid for rid in d08 if rid in d4b]
    acc08 = {rid: float(max(d08[rid]["probs"], key=d08[rid]["probs"].get) == d08[rid]["label"])
             for rid in common}
    acc4b = {rid: float(max(d4b[rid]["probs"], key=d4b[rid]["probs"].get) == d4b[rid]["label"])
             for rid in common}
    flat08 = float(np.mean(list(acc08.values())))
    flat4b = float(np.mean(list(acc4b.values())))
    rows = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = int(round(frac * len(common)))
        esc = set(sorted(common, key=lambda kk: max(d08[kk]["probs"].values()))[:k])
        corr = [acc4b[rid] if rid in esc else acc08[rid] for rid in common]
        rows.append({"escalate_frac": frac, "acc": round(float(np.mean(corr)), 4),
                     "cost_units_08b": round(1.0 + frac * 4.0, 2),
                     "random_routing_acc": round((1 - frac) * flat08 + frac * flat4b, 4)})
    out[src] = {"n": len(common), "flat_08b": round(flat08, 4), "flat_4b": round(flat4b, 4),
                "gap_pp": round(100 * (flat4b - flat08), 1), "curve": rows}
    print(f"== {src} (0.8B {flat08:.4f} / 4B {flat4b:.4f}, gap {100*(flat4b-flat08):.1f}pp)")
    for r in rows:
        gate_gain = r["acc"] - r["random_routing_acc"]
        print(f"  r={r['escalate_frac']:.2f}: cascade={r['acc']:.4f} "
              f"random={r['random_routing_acc']:.4f} gate_gain={gate_gain:+.4f} "
              f"@{r['cost_units_08b']:.2f}u")

json.dump(out, open(OUT / "cascade_curves.json", "w"), indent=1)
print("saved cascade_curves.json")
