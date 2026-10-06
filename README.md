# Menu-Operation Probe Batteries and Confidence Cascades for Prefill-Only Decision Models

**Shrink the menu, keep the answer — and pay only for the questions the small model
gets wrong.**

Code for the two main experiment lines of our paper on **prefill-only decision
models**: models that answer by reading out a full probability distribution over a
label menu in a single forward pass, instead of generating text.

---

## Demo: escalate, don't re-ask

A confidence cascade costs almost nothing to build for a prefill-only decision model:
the small model answers over the full menu; the *least-confident fraction* of records
is re-answered by the larger model. Because that second pass is exactly the large
model's flat pass, the whole cascade needs only **two flat passes** — no extra
engineering, no thresholding sweeps at deployment.

Real numbers from `cascade_analysis.py` (kev-0.8b → kev-4b, HWU-64 test, cost in
0.8B-forward units):

| escalation | cost(model sizes) | cascade acc | random routing @ same cost |
|---|---|---|---|
| r = 0 (kev-0.8B) (small model only) | 1.0u | 61.9 | 61.9 |
| **r = 0.25** | **2.0** | **71.6** | 64.4 |
| r = 0.50 | 3.0 | 72.6 | 66.9 |
| r = 1.0 (kev-4B) (large model only) | 5.0u | 71.9 | 71.9 |

Escalating just the 25% least-confident records recovers **96% of the large model's
accuracy at 40% of the extra cost** — and beats random routing at the same budget by
**+7.2pp**. The value is in *knowing where the small model is weak*: confidence is a
good signal of weakness, so the same budget spent on random records buys far less.
This holds across all five evaluation sources (the paper's §4: escalation dominates
self-re-asking at every cost point).

Reproduce it yourself:

```bash
export KEV_ROOT=/path/to/kev-repo
python scripts/cascade_kev_pass1.py hwu64     # large-model flat pass (second pass)
python scripts/cascade_analysis.py            # -> results/cascade/cascade_curves.json
```

Code for reproducing the two main experiment lines of the paper:

1. **Menu-operation probe batteries** (restriction baseline / readout consistency).
   A decision model first answers over the *full* label menu (flat pass-1). The pass-1
   distribution is then *restricted* to a smaller candidate menu and renormalized; the
   restriction prediction is compared against a second pass actually run with that menu.
   The menu operations (cells) are:
   - `random15` — uniform rivals + gold, shuffled (restriction-baseline cell)
   - `random15_free` — 15 uniform labels, gold NOT forced (gold-free deployment cell)
   - `avoid15` — confusion-avoidant rivals + gold, rival weights ~ 1/(p1+0.02)
   - `avoid15_free` — confusion-avoidant, gold NOT forced
   - `self15_clean` — pass-1 top rivals + gold, original task instruction
   - `self15_verify` — same menus, comparison-style wording ("Which of these C options
     matches best?") to expose presentation effects
   - `self2_clean` — top-1 rival + gold
2. **Confidence cascades.** A small decision model answers first over the full menu;
   the lowest-confidence fraction of records is escalated to a larger model. Because the
   second pass is the larger model's flat pass, the whole cascade needs only two flat
   passes. Curves are reported against a random-routing control at matched cost.

## Repository layout

```
scripts/
  build_sources.py      rebuild data/<source>_test.jsonl (+ labels) from public datasets
  battery_kev.py        menu battery, kev family via local predictor (GPU)
  battery_jev.py        menu battery, Open-Jev family via vllm-jev HTTP endpoint
  battery_analysis.py   restriction-baseline residuals + frozen gates (G-A1/A2/A3)
  cascade_kev_pass1.py  kev-4b flat pass-1 per source (cascade second pass)
  cascade_analysis.py   cascade curves + random-routing control (offline)
  cascade_tasks.py      two-stage small->large cascade over HTTP on 500-record tasks
  serve_model.sh        serve one model with vllm-jev and run a script against it
data/                   evaluation records (five wide-label-space sources, full test)
results/                created at runtime; all JSON outputs land here
```

## Requirements

- Python 3.10+, `numpy` (see `requirements.txt`)
- The **kev** package (provides `kev.model.ContextOverflow`,
  `kev.predictors.LocalPredictor`); point `KEV_ROOT` at a local checkout of the kev
  repository.
- For the Open-Jev / HTTP experiments: a **vllm-jev** server (endpoint `POST
  /v1/systemone` returning per-question `probabilities`), plus `curl`.

## Models used (Hugging Face repos)

| Role | Repo |
|---|---|
| kev family, small (cascade stage 1, main battery) | `jaredpalmer/kev-0.8b` |
| kev family, large (cascade stage 2) | `jaredpalmer/kev-4b` |
| Open-Jev family (cross-family battery, HTTP cascade) | `ZefanCai/Open-Jev-2B` |

Model weights are downloaded by the respective inference stacks; no weights are
included here.

## Reproduction guide

All commands run from the repository root. GPU scripts honor `CUDA_VISIBLE_DEVICES`.

### 0. Data

The five test files and label lists are shipped in `data/`. To rebuild them from the
original public datasets (GoEmotions-27, MASSIVE-60, HWU-64, AmazonToys-60, SNIPS-7):

```bash
python scripts/build_sources.py
```

### 1. Menu-operation battery, kev family (local predictor)

```bash
export KEV_ROOT=/path/to/kev-repo
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/battery_kev.py $s
done
python scripts/battery_analysis.py      # residuals + gates -> results/battery_analysis.json
```

Each run writes `results/battery_kev/<source>_pass1.json` (flat full-menu
distributions) and one `<source>_<cell>.json` per menu cell. `battery_analysis.py`
computes, per (model, source, cell): measured accuracy, the restriction prediction,
the residual, and (for the gold-free cells) the gold hit rate and conditional
accuracy; it then evaluates the frozen gates G-A1 (restriction baseline, |residual|
<= 2pp), G-A2 (avoid >= random - 1pp) and the G-A3 self2 sign table.

### 2. Menu-operation battery, Open-Jev family (vllm-jev HTTP)

Start the server for `ZefanCai/Open-Jev-2B` (any host; endpoint configurable via
`BATTERY_URL`, default `http://127.0.0.1:8796`), then either run the battery directly:

```bash
BATTERY_URL=http://127.0.0.1:8796 python scripts/battery_jev.py goemotions27
BATTERY_URL=http://127.0.0.1:8796 python scripts/battery_jev.py massive60
```

or let `serve_model.sh` manage the server lifecycle:

```bash
export VLLM_JEV_VENV=/path/to/vllm-jev-venv
export VLLM_JEV_WORKSPACE=/path/to/vllm-jev/workspace
bash scripts/serve_model.sh 0 8796 ZefanCai/Open-Jev-2B scripts/battery_jev.py goemotions27
```

`battery_analysis.py` picks the jev results up automatically (prefix `jev2b_`).

### 3. Confidence cascade, kev-0.8b -> kev-4b

```bash
export KEV_ROOT=/path/to/kev-repo
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/cascade_kev_pass1.py $s        # kev-4b flat pass-1 (second pass)
done
python scripts/cascade_analysis.py               # -> results/cascade/cascade_curves.json
```

Cost units are kev-0.8b = 1, kev-4b = 5 (parameter ratio). For each escalation
fraction r in {0, .25, .5, .75, 1} the curve reports cascade accuracy, cost, and the
random-routing control accuracy at matched cost; the gate gain
(cascade - random-routing) is the quantity of interest.

### 4. Confidence cascade over HTTP (small -> large, hard tasks)

With the small model served at `CASCADE_URL`:

```bash
python scripts/cascade_tasks.py flat      # small-model flat passes (goemotions27, amzntoys60)
# switch the server to the large model, then:
python scripts/cascade_tasks.py flat2     # large-model flat passes
python scripts/cascade_tasks.py curve     # -> results/cascade_tasks/cascade_task_curves.json
```

or via the helper (runs one server + one stage):

```bash
bash scripts/serve_model.sh 0 8796 <small-model-hf-id> scripts/cascade_tasks.py flat
bash scripts/serve_model.sh 0 8796 <large-model-hf-id> scripts/cascade_tasks.py flat2
python scripts/cascade_tasks.py curve
```

### Notes

- Menu seeds are derived deterministically from `crc32(key)` of the source/cell/record
  id, so all cells are exactly reproducible across machines and model families.
- Narrow label spaces (snips7, 7 labels) cap the menu size C at the label-set size;
  cells are compared at matched C.
- In gold-free cells (`*_free`), records whose sampled menu misses gold are scored 0
  by construction (deployment semantics) and skip the model call; the hit rate and
  gold-conditional accuracy are reported alongside.
- Everything is resumable: existing per-cell output files are skipped on re-run.
