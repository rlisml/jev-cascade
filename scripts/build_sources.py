"""Build the five wide-label-space evaluation sources in the record format used here.

Sources (full test split where available):
  goemotions27 : google-research-datasets/go_emotions test, single-label non-neutral
                 subset (27 labels)
  massive60    : SetFit/amazon_massive_intent_en-US test (60 intents)
  hwu64        : FastFit/hwu_64 test (64 intents)
  amzntoys60   : milistu/Amazon_Toys_and_Games_2014 metadata, leaf product categories,
                 top-60 leaves with >=80 products, 1200 sampled records
  snips7       : benayas/snips test (7 categories; narrow contrast, recorded honestly)
Record format: {"rid", "state" (plain text), "qid", "instr", "label"}.
Outputs: data/<source>_test.jsonl + data/<source>_labels.json
"""
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"


def save(source, rows):
    labels = sorted({r["label"] for r in rows})
    with (OUT / f"{source}_test.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / f"{source}_labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    print(f"{source}: {len(rows)} records, {len(labels)} labels", flush=True)


# ---- goemotions27 ----
ds = load_dataset("google-research-datasets/go_emotions", split="test")
names = ds.features["labels"].feature.names
rows = []
for n, ex in enumerate(ds):
    if len(ex["labels"]) != 1 or ex["labels"][0] == 27:  # 27 = neutral
        continue
    rows.append({"rid": f"goemotions27/test/{n}", "state": ex["text"], "qid": "emotion",
                 "instr": "Which emotion does this text express?",
                 "label": names[ex["labels"][0]]})
save("goemotions27", rows)

# ---- massive60 ----
ds = load_dataset("SetFit/amazon_massive_intent_en-US", split="test")
rows = [{"rid": f"massive60/test/{ex['id']}", "state": ex["text"], "qid": "intent",
         "instr": "Which assistant intent does this request match?",
         "label": ex["label_text"]} for ex in ds]
save("massive60", rows)

# ---- hwu64 ----
ds = load_dataset("FastFit/hwu_64", split="test")
rows = [{"rid": f"hwu64/test/{n}", "state": ex["text"], "qid": "intent",
         "instr": "Which assistant intent does this request match?",
         "label": ex["label"]} for n, ex in enumerate(ds)]
save("hwu64", rows)

# ---- amzntoys60 ----
ds = load_dataset("milistu/Amazon_Toys_and_Games_2014", "metadata", split="train")
leaf_cnt = Counter()
recs = []
for ex in ds:
    c = ex.get("categories")
    if not (isinstance(c, list) and c and isinstance(c[0], list) and c[0]):
        continue
    leaf = c[0][-1]
    if leaf in ("Toys & Games",):  # generic root-ish leaf, not a class
        continue
    leaf_cnt[leaf] += 1
    title = (ex.get("title") or "").strip()
    desc = (ex.get("description") or "").strip()
    if not title:
        continue
    recs.append({"state": f"{title}. {desc}"[:1500], "leaf": leaf})
top = [l for l, n in leaf_cnt.most_common(60) if n >= 80][:60]
pool = [r for r in recs if r["leaf"] in top]
rng = random.Random(0)
rng.shuffle(pool)
pool = pool[:1200]
rows = [{"rid": f"amzntoys60/test/{n}", "state": r["state"], "qid": "category",
         "instr": "Which product category does this toy or game belong to?",
         "label": r["leaf"]} for n, r in enumerate(pool)]
save("amzntoys60", rows)

# ---- snips7 ----
ds = load_dataset("benayas/snips", split="test")
rows = [{"rid": f"snips7/test/{n}", "state": ex["text"], "qid": "intent",
         "instr": "Which assistant intent does this request match?",
         "label": ex["category"]} for n, ex in enumerate(ds)]
save("snips7", rows)

print("build_sources done", flush=True)
