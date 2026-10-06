"""Analysis of the menu-operation batteries: restriction-baseline residuals and gates.

Restriction baseline: per record, restrict the pass-1 full distribution to the cell's
menu, renormalize; argmax == gold -> predicted correct. residual = predicted_acc -
measured_acc (positive = the menu intervention hurts vs the readout-consistency
prediction).

Gates (frozen before running):
  G-A1 restriction baseline holds on new sources: |residual| <= 2pp for random15 and
       avoid15 on >= 9/10 (source x model) cells
  G-A2 avoid15 >= random15 - 1pp on >= 8/10 sources (else avoid is dataset-specific)
  G-A3 self2 sign table vs flat per source (record signs honestly)
Outputs: results/battery_analysis.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P_KEV = ROOT / "results" / "battery_kev"
P_JEV = ROOT / "results" / "battery_jev"
SOURCES = ["goemotions27", "massive60", "hwu64", "amzntoys60", "snips7"]
PREFIXES = {"kev0.8b": (P_KEV, ""), "jev2b": (P_JEV, "jev2b_")}
MENU_CELLS = ["random15", "random15_free", "avoid15", "avoid15_free",
              "self15_clean", "self15_verify", "self2_clean"]


def argmax_key(dist):
    return max(dist, key=dist.get)


def restriction_pred(probs, menu, gold):
    sub = {o: probs.get(o, 0.0) for o in menu}
    z = sum(sub.values())
    if z <= 0:
        return 0.0
    return float(max(sub, key=sub.get) == gold)


def gold_in(measured):
    hit = np.mean([r["gold_in_menu"] for r in measured])
    cond = [r["correct"] for r in measured if r["gold_in_menu"]]
    return {"hit_rate": round(float(hit), 4),
            "acc": round(float(np.mean([r["correct"] for r in measured])), 4),
            "acc_given_gold_in_menu": round(float(np.mean(cond)), 4) if cond else None}


def load(base, prefix, source, suffix):
    f = base / f"{prefix}{source}_{suffix}.json"
    if not f.exists():
        return None
    return json.load(open(f))


report = {"per_cell": {}, "gates": {}}
for model_key, (base, prefix) in PREFIXES.items():
    for source in SOURCES:
        p1 = load(base, prefix, source, "pass1")
        if p1 is None:
            continue
        dist = p1["dists"]
        fcell = load(base, prefix, source, "flat")
        if fcell is not None:
            flat_acc = float(np.mean([r["correct"] for r in fcell["rows"]]))
        else:  # kev battery stores pass-1 only for flat; derive the cell from it
            flat_acc = float(np.mean([argmax_key(d["probs"]) == d["label"]
                                      for d in dist.values()]))
        for cell in MENU_CELLS:
            c = load(base, prefix, source, cell)
            if c is None:
                continue
            measured = float(np.mean([r["correct"] for r in c["rows"]]))
            preds, golds = [], []
            for r in c["rows"]:
                d = dist.get(r["rid"])
                if d is None:
                    continue
                preds.append(restriction_pred(d["probs"], r["menu"], d["label"]))
                golds.append(r["correct"])
            residual = float(np.mean(preds) - np.mean(golds))
            row = {"n": c["n"], "acc": round(measured, 4),
                   "restriction_pred": round(float(np.mean(preds)), 4),
                   "residual": round(residual, 4)}
            if "_free" in cell:
                row.update(gold_in(c["rows"]))
            report["per_cell"][f"{model_key}|{source}|{cell}"] = row
        report["per_cell"][f"{model_key}|{source}|flat"] = {"acc": round(flat_acc, 4)}

# ---- gates ----
cells = report["per_cell"]
g_a1, g_a2, g_a3 = [], [], {}
for model_key in PREFIXES:
    for source in SOURCES:
        rr = cells.get(f"{model_key}|{source}|random15")
        av = cells.get(f"{model_key}|{source}|avoid15")
        if rr is None or av is None:
            continue
        g_a1.append({"cell": f"{model_key}|{source}", "random15_resid": rr["residual"],
                     "avoid15_resid": av["residual"],
                     "hold": abs(rr["residual"]) <= 0.02 and abs(av["residual"]) <= 0.02})
        g_a2.append({"source": f"{model_key}|{source}", "avoid15": av["acc"],
                     "random15": rr["acc"], "hold": av["acc"] >= rr["acc"] - 0.01})
        s2 = cells.get(f"{model_key}|{source}|self2_clean")
        fl = cells.get(f"{model_key}|{source}|flat")
        if s2 and fl:
            g_a3[f"{model_key}|{source}"] = {"self2_acc": s2["acc"], "flat_acc": fl["acc"],
                                             "self2_minus_flat": round(s2["acc"] - fl["acc"], 4)}

report["gates"]["G_A1_restriction_baseline"] = {
    "rule": "|residual| <= 2pp on >= 9/10 (source x model) cells",
    "n_hold": sum(x["hold"] for x in g_a1), "n": len(g_a1), "cells": g_a1}
report["gates"]["G_A2_avoid_vs_random"] = {
    "rule": "avoid15 >= random15 - 1pp on >= 8/10",
    "n_hold": sum(x["hold"] for x in g_a2), "n": len(g_a2), "cells": g_a2}
report["gates"]["G_A3_self2_sign_table"] = {
    "rule": "record signs honestly",
    "cells": g_a3}

json.dump(report, open(ROOT / "results" / "battery_analysis.json", "w"), indent=1)

print("== per-cell (acc / restriction_pred / residual) ==")
for k, v in report["per_cell"].items():
    if "residual" in v:
        extra = (f" hit={v['hit_rate']:.2f} cond_acc={v['acc_given_gold_in_menu']}"
                 if "hit_rate" in v else "")
        print(f"{k:48s} acc={v['acc']:.4f} pred={v['restriction_pred']:.4f} "
              f"resid={v['residual']:+.4f}{extra}")
    else:
        print(f"{k:48s} acc={v['acc']:.4f} (flat)")
print("\n== gates ==")
for g, d in report["gates"].items():
    if "n_hold" in d:
        print(f"{g}: {d['n_hold']}/{d['n']} hold  [{d['rule']}]")
print("saved battery_analysis.json")
