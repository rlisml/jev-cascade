# Inference-time compute strategy for (Jev-like) Prefill-Only Decision Models with Confidence Cascades:

**Let a 0.8B model answer like a 4B: re-run only the 25% questions it is least sure
about — two flat passes, 40% of the extra cost, 96% of the accuracy.**

**Against a 9B model, the same recipe at half the 9B's cost lands within 1pp of it —
and beats random routing at that budget by +8pp.**

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

| escalation | cost | cascade acc | random routing @ same cost |
|---|---|---|---|
| r = 0 (small model only) | 1.0 | 61.9 | 61.9 |
| **r = 0.25** | **2.0** | **71.6** | 64.4 |
| r = 0.50 | 3.0 | 72.6 | 66.9 |
| r = 1.0 (large model only) | 5.0 | 71.9 | 71.9 |

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

## Demo: the same recipe against a 9B

Scale the second stage up to kev-9b and the story gets stronger. Pooled over 3,500
records (seven public sources, 500 test records each; cost in 0.8B-forward units,
kev-9b = 11.25 by parameter ratio):

| escalation | cost | cascade acc | random routing @ same cost |
|---|---|---|---|
| r = 0 (small model only) | 1.00 | 70.9 | 70.9 |
| r = 0.25 | 3.56 | 81.5 | 75.4 |
| **r = 0.50** | **6.12** | **87.9** | 79.8 |
| r = 1.0 (9B everywhere) | 11.25 | 88.7 | 88.7 |

The r = 0.5 cascade **matches the 9B model (87.9 vs 88.7) at 54% of its cost**, and
beats random routing at the same budget by **+8.1pp**. The biggest wins are the
knowledge-heavy sources where the small model is far behind — on ARC the cascade moves
61.6 → 67.8 (r = 0.1) → 75.6 (r = 0.25) toward the 9B's 94.4.

**How to get this data.** The seven sources are the kev package's built-in public
datasets — MMLU, ARC, OpenBookQA, CommonsenseQA, SciQ, AG News, and DBpedia-14 — built
into the same record format as `data/` by the kev data builder
(`kev.data.build`, 500 test records per source, seed 0). With those records on disk,
the pipeline is identical to the 4B cascade above, with `"jaredpalmer/kev-9b"` as the
second stage:

```bash
export KEV_ROOT=/path/to/kev-repo
python scripts/cascade_kev_pass1.py mmlu      # per source, with MODEL_STAGE2 override
python scripts/cascade_analysis.py
```

(The shipped `data/` sources power the 4B demo; the seven-source build powers this one.
Both use the identical record format, so `cascade_kev_pass1.py` takes either — set the
second-stage model via the `MODEL_STAGE2` environment variable, default
`jaredpalmer/kev-4b`.)

## Demo: predict a menu change without running the model

For prefill-only models there is a sharp baseline prediction for what happens when you
*shrink the candidate menu* before a second pass: restrict the pass-1 distribution to
the menu, renormalize, read off the answer. When measurement matches prediction, a menu
redesign is a **zero-labeling intervention simulator** — you can evaluate deployment
menus offline. The residual, where it appears, is a fingerprint of presentation
effects and confusable labels. Seven menu operations probe this (`battery_kev.py` /
`battery_jev.py`), and `battery_analysis.py` scores the residuals against three
pre-registered gates.

```
                     ┌─────────────────────────────────────────┐
                     │ pass 1: full menu (150 labels, say)     │
                     │ one forward pass → full distribution p1 │
                     └───────────────────┬─────────────────────┘
                                         │
              ┌──────────────────────────┴───────────────────────────┐
              │ menu operations (probe battery)                      │ confidence cascade
              │ random15 / avoid15 / self15 / self2                  │ low-confidence fraction
              │ predict accuracy from p1 alone                       │ escalated to the large model
              │ residual = what the model "feels" about the menu     │ accuracy/cost Pareto curve
              └──────────────────────────────────────────────────────┘
```

## What's inside

| Script | What it does |
|---|---|
| `build_sources.py` | Rebuilds the five evaluation datasets from public Hugging Face sources |
| `battery_kev.py` | Menu-operation battery, kev family, local predictor (GPU) |
| `battery_jev.py` | Menu-operation battery, Open-Jev family, over HTTP |
| `battery_analysis.py` | Restriction-baseline residuals + the three pre-registered gates |
| `cascade_kev_pass1.py` | Large-model (kev-4b) flat pass — the cascade's second stage |
| `cascade_analysis.py` | Cascade curves vs. a random-routing control |
| `cascade_tasks.py` | Two-stage small→large cascade over HTTP on hard tasks |
| `serve_model.sh` | Serves one model with vllm-jev and runs a script against it |

The **menu operations** (each is one "cell" of the probe battery):

| Cell | Menu construction | What it probes |
|---|---|---|
| `random15` | 14 uniform rivals + gold, shuffled | does the restriction baseline hold? |
| `random15_free` | 15 uniform labels, gold **not** forced | gold-free deployment reality |
| `avoid15` | rivals sampled ∝ 1/(p₁+0.02) + gold | can we dodge confusables? |
| `avoid15_free` | avoidance sampling, gold not forced | same, deployment-realistic |
| `self15_clean` | pass-1 top rivals + gold | the model's own hardest rivals |
| `self15_verify` | same menus, "Which of these C options matches best?" | presentation effects |
| `self2_clean` | top-1 rival + gold | forced binary choice |

All menus are seeded deterministically from `crc32` of the source/cell/record id, so
every cell is bit-for-bit reproducible across machines and model families.

## Quickstart

```bash
pip install -r requirements.txt
export KEV_ROOT=/path/to/kev-repo        # the kev package checkout

# 1) probe battery + restriction-baseline analysis (kev family, GPU)
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/battery_kev.py $s
done
python scripts/battery_analysis.py

# 2) confidence cascade: kev-0.8b -> kev-4b
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/cascade_kev_pass1.py $s
done
python scripts/cascade_analysis.py
```

For the Open-Jev (HTTP) side of the story:

```bash
export VLLM_JEV_VENV=/path/to/vllm-jev-venv
export VLLM_JEV_WORKSPACE=/path/to/vllm-jev/workspace
bash scripts/serve_model.sh 0 8796 ZefanCai/Open-Jev-2B scripts/battery_jev.py goemotions27
# then: battery_analysis.py picks the jev results up automatically
```

## Models (Hugging Face repos)

| Role | Repo |
|---|---|
| kev family, small — battery + cascade stage 1 | `jaredpalmer/kev-0.8b` |
| kev family, large — cascade stage 2 (default) | `jaredpalmer/kev-4b` |
| kev family, larger — cascade stage 2 alternative | `jaredpalmer/kev-9b` |
| Open-Jev family — cross-family battery + HTTP cascade | `ZefanCai/Open-Jev-2B` |

No model weights are included; the inference stacks download them from the repos above.

## Data

Five wide-label-space sources (full test splits) ship in `data/`:

| Source | Records | Labels |
|---|---|---|
| `goemotions27` | 2,984 | 27 emotions |
| `massive60` | 2,974 | 59 assistant intents |
| `hwu64` | 1,076 | 64 assistant intents |
| `amzntoys60` | 1,200 | 60 product categories |
| `snips7` | 1,400 | 7 intents (narrow contrast) |

`python scripts/build_sources.py` rebuilds all of them byte-identically from the
original public datasets (GoEmotions, MASSIVE, HWU-64, Amazon Toys & Games, SNIPS).

## Reproduction guide (full)

All commands run from the repository root. GPU scripts honor `CUDA_VISIBLE_DEVICES`.
Everything is resumable — existing per-cell output files under `results/` are skipped
on re-run.

### 1. Menu-operation battery, kev family (local predictor)

```bash
export KEV_ROOT=/path/to/kev-repo
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/battery_kev.py $s
done
python scripts/battery_analysis.py        # -> results/battery_analysis.json
```

Each source produces `results/battery_kev/<source>_pass1.json` (flat full-menu
distributions) plus one `<source>_<cell>.json` per menu cell. `battery_analysis.py`
then computes, per (model, source, cell): measured accuracy, the restriction
prediction, the residual, and — for the gold-free cells — the gold hit rate and
gold-conditional accuracy. It evaluates three gates frozen before the runs:

- **G-A1** — restriction baseline: |residual| ≤ 2pp on ≥ 9/10 (source × model) cells
- **G-A2** — avoid ≥ random − 1pp (otherwise avoidance is dataset-specific)
- **G-A3** — self2 vs flat sign table, recorded honestly per source

### 2. Menu-operation battery, Open-Jev family (vllm-jev HTTP)

Start the server for `ZefanCai/Open-Jev-2B` (endpoint configurable via `BATTERY_URL`,
default `http://127.0.0.1:8796`):

```bash
BATTERY_URL=http://127.0.0.1:8796 python scripts/battery_jev.py goemotions27
BATTERY_URL=http://127.0.0.1:8796 python scripts/battery_jev.py massive60
```

or let `serve_model.sh` manage the server lifecycle (as in the Quickstart). The
analysis script picks the jev results up automatically (prefix `jev2b_`).

### 3. Confidence cascade, kev-0.8b → kev-4b (or kev-9b)

```bash
export KEV_ROOT=/path/to/kev-repo
for s in goemotions27 massive60 hwu64 amzntoys60 snips7; do
  python scripts/cascade_kev_pass1.py $s        # kev-4b flat pass-1 (second pass)
done
# second stage kev-9b instead (any source set, incl. the seven-source build):
MODEL_STAGE2=jaredpalmer/kev-9b python scripts/cascade_kev_pass1.py hwu64
python scripts/cascade_analysis.py               # -> results/cascade/cascade_curves.json
```

Cost units: kev-0.8b = 1, kev-4b = 5, kev-9b = 11.25 (parameter ratios). For each
escalation fraction r ∈ {0, .25, .5, .75, 1} the curve reports cascade accuracy, cost,
and the random-routing control at matched cost. The quantity of interest is the gate
gain (cascade − random routing): the value of *knowing where the small model is weak*.

### 4. Confidence cascade over HTTP (small → large, hard tasks)

```bash
python scripts/cascade_tasks.py flat      # small model, 500 records per task
# switch the server to the large model, then:
python scripts/cascade_tasks.py flat2     # large model
python scripts/cascade_tasks.py curve     # -> results/cascade_tasks/cascade_task_curves.json
```

or via the helper:

```bash
bash scripts/serve_model.sh 0 8796 <small-model-hf-id> scripts/cascade_tasks.py flat
bash scripts/serve_model.sh 0 8796 <large-model-hf-id> scripts/cascade_tasks.py flat2
python scripts/cascade_tasks.py curve
```

## Implementation notes

- Narrow label spaces (snips7, 7 labels) cap the menu size C at the label-set size;
  cells are compared at matched C.
- In gold-free cells (`*_free`), records whose sampled menu misses gold are scored 0
  by construction (deployment semantics) and skip the model call; hit rate and
  gold-conditional accuracy are reported alongside.
- The HTTP endpoint contract is a single `POST /v1/systemone` returning per-question
  `probabilities` — any serving stack with that contract works in place of vllm-jev.
