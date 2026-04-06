# autoresearch-sleep

Autonomous research loop for sleep stage classification on single-channel EEG.

## Core principle

**Power comes through understanding.** Every improvement should be grounded in understanding *why* the current approach is limited, not just *that* it is limited. Blind hyperparameter search has diminishing returns. Breakthroughs come from illuminating something about the data, the task, or the model's failure modes — and then acting on that understanding.

When stuck, the answer is almost never "try another hyperparameter." It's "I don't understand something yet." Find out what that is.

## Setup

When the user says to kick off an experiment, do ALL of the following setup steps without pausing for confirmation, then immediately proceed to the experiment loop:

1. **Pick a run tag**: use today's date (e.g. `mar27`). The branch `autoresearch-sleep/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-sleep/<tag>` from the current branch.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — data loading, train/val split, dataloader, evaluation. Contains the fixed evaluation harness. **Do not modify the `evaluate` function.**
   - `train.py` — model architecture, optimizer, training loop.
   - `phases/02_model_development_and_evaluation/general_model_structure.md` — the target model architecture. Follow this structure.
   - `phases/01_data_exploration_and_literature_review/components/` — reference library of techniques. Read all files here for ideas.
4. **Verify data exists**: Check that the dataset path referenced in `prepare.py` exists and is loadable. If not, tell the human and stop.
5. **Understand the data before you touch the model** (see "Understanding phase" below).
6. **Initialize results.tsv**: Create `results.tsv` in the campaign directory with just the header row. The baseline will be recorded after the first run.
7. **Immediately run the baseline** — do NOT ask the user for confirmation. Go straight into the experiment loop below.

### Campaign directory

Each run creates a new campaign folder:

```
phases/02_model_development_and_evaluation/campaigns/<run_tag>/
```

All experiment artifacts for the run live here: `results.tsv`, `diagnosis.md`, `run.log`. The `train.py` and `prepare.py` files live at the project root.

## Task description

**Dataset**: Sleep-EDF (processed). Single-channel EEG (Fpz-Cz), 30-second epochs at 100 Hz (3000 samples per epoch), 5 AASM sleep stages (Wake, N1, N2, N3, REM). Data is pre-segmented into epochs. The dataset files are **read-only** — do not modify them.

**Classes**: 5 (Wake=0, N1=1, N2=2, N3=3, REM=4). Highly imbalanced (~69% Wake, 17% N2, 6% REM, 5% N1, 3% N3).

**Classification constraint**: The model classifies the **last window** in its input. It may use one epoch (one-to-one) or multiple consecutive past epochs (many-to-one) as context, but it **cannot use future epochs**. The input is causal. The evaluation metric is always computed on the classification of the final epoch.

### Data splits

The data is split into **train** and **val** (validation), by subject — entire recording sessions go into one partition, never split across partitions.

```
data/
├── train/    — training data. The model trains on this.
└── val/      — validation data. Used for keep/discard decisions.
```

The model trains on `data/train/` and evaluates on `data/val/`. The `evaluate` function in `prepare.py` runs on val data only.

**Metrics** (in order of importance):
1. **Accuracy** (primary, higher is better)
2. **Cohen's kappa** (primary, higher is better)
3. **Macro F1** (diagnostic, higher is better)
4. Per-class F1 / precision / recall (diagnostic)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `python train.py`.

**What you CAN modify:**
- `train.py` — model architecture, optimizer, hyperparameters, training loop, batch size, model size, loss function, augmentation, scheduling, etc.
- Data loading in `prepare.py` — how epochs are read, windowed, grouped, or preprocessed before being fed to the model. You can change the input representation: re-window, combine multiple consecutive epochs, reshape, apply transforms. The only constraint is causality (no future epochs).
- Add new Python files if needed (e.g., `model.py`, `augmentations.py`).

**What you CANNOT modify:**
- The `evaluate` function in `prepare.py`. It is the ground truth metric.
- The dataset files themselves. The `.npz` files are read-only.
- Dependencies. Only use what's already in `pyproject.toml`.

**The goal is simple: get the highest accuracy and F1Score.** Since the time budget is fixed, everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function, the input representation. The only constraints are: (1) the code runs without crashing within the time budget, (2) classification is causal (no future epochs), and (3) the evaluate function is untouched.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**Reference material**: Before designing experiments, consult:
- `phases/02_model_development_and_evaluation/general_model_structure.md` — follow this general architecture.
- `phases/01_data_exploration_and_literature_review/components/` — menu of techniques to pull from.

**The first run**: Always establish the baseline first by running the training script as-is.

## Understanding phase

Before your first experiment (after the baseline) and whenever you hit a plateau, invest time in *understanding* the problem. Write findings to `diagnosis.md` in the campaign directory.

### Initial data exploration (before first non-baseline experiment)

Examine the raw data:
- What are the shapes? What does each dimension mean physically?
- How are epochs ordered within subjects? Are consecutive epochs temporally contiguous?
- What's the sampling rate? What time duration does each epoch represent?
- What distinguishes the classes? Compute basic statistics per class.
- **Separate what's in the data from what's in the pipeline.** The `.npz` files reflect preprocessing choices already made. The pipeline in `prepare.py` makes its own choices on top. Map the full chain from raw data to model input. At each step distinguish between *properties of the data* (immutable) and *design choices* (things the pipeline could do differently). The biggest gains often come from changing the representation, not the model.

### Plateau diagnosis (mandatory when stuck)

When `experiments_since_improvement` reaches **10**, **stop experimenting and start diagnosing.** Do NOT skip this. Do NOT run "just one more quick experiment."

After completing diagnosis, **update `diagnosis.md`** with findings. The file should always reflect current state — current best accuracy/kappa, current gaps, structural near-misses. Delete outdated analysis.

**Error analysis:**
- Compute per-class F1, not just macro F1. Which class is hardest? (Expect N1 to be worst.)
- Look at confidence of errors — is the model uncertain or confidently wrong?
- Check train vs. val metrics. If train >> val, overfitting. If train ~ val ~ plateau, the input representation is the bottleneck.
- Examine the confusion matrix. Which classes get confused with each other?

**Gap analysis — the most important step:**
For every property of the data you have discovered, ask:
1. **Does the current pipeline exploit this property?** If the data has structure that the pipeline discards, that is a gap — and gaps are where the biggest improvements hide.
2. **What experiment would close this gap?** Design a concrete change. Propose at least one experiment for every gap.

Pattern: *property -> gap -> experiment*.

**Assumption audit:**
List every assumption in the current pipeline. Split into:
1. **Representation assumptions** — what the model sees: input shape, window size, how many epochs of context, what preprocessing is applied. These define the ceiling of what the model *can* learn. Highest leverage.
2. **Model assumptions** — how the model processes its input: architecture, capacity, receptive field, training procedure.

For each, ask: "If this assumption is wrong, what would I see in the errors?" Then check.

**Revisit log:**
Structural changes should be revisited when the baseline has improved significantly since they were last tested. Keep a short list of "structural near-misses" in diagnosis.md and **retest the top 2 after every significant improvement in best accuracy** before pursuing new ideas.

## Output format

The training script prints a summary when finished:

```
---
accuracy:          0.8500
kappa:             0.7800
macro_f1:          0.7200
class_0_f1:       0.9100  (p=0.920 r=0.900)
class_1_f1:       0.4500  (p=0.500 r=0.410)
class_2_f1:       0.7800  (p=0.770 r=0.790)
class_3_f1:       0.6500  (p=0.680 r=0.623)
class_4_f1:       0.7100  (p=0.700 r=0.720)
training_seconds:  600.1
total_seconds:     625.9
peak_vram_mb:      12345.6
total_samples_M:   12.3
num_steps:         953
num_params_M:      3.1
depth:             4
```

Extract key metrics:

```
grep "^accuracy:\|^kappa:\|^macro_f1:\|^class_.*_f1:\|^peak_vram_mb:" run.log
```

**Primary optimization targets**: `accuracy` and `kappa` (higher is better). Keep/discard decisions are based on accuracy first, kappa as tiebreaker.

**Diagnostic metrics**: `macro_f1` and per-class F1/precision/recall. Use these to understand *where* the model is struggling. If class_1_f1 (N1) is much lower than others, that's expected but still a signal. If precision >> recall for a class, the model is being conservative.

## Logging results

Log results to `results.tsv` in the campaign directory (tab-separated, NOT comma-separated).

Header row and 7 columns:

```
commit	accuracy	kappa	macro_f1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. accuracy (e.g. 0.8500) — use 0.0 for crashes
3. kappa (e.g. 0.7800) — use 0.0 for crashes
4. macro_f1 (e.g. 0.7200) — use 0.0 for crashes
5. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	accuracy	kappa	macro_f1	memory_gb	status	description
a1b2c3d	0.8500	0.7800	0.7200	12.0	keep	baseline
b2c3d4e	0.8620	0.7950	0.7400	12.2	keep	increase depth to 6 (STRUCTURAL)
c3d4e5f	0.8480	0.7750	0.7100	12.0	discard	switch to GeLU activation (TUNING)
d4e5f6g	0.0	0.0	0.0	0.0	crash	double model width OOM (STRUCTURAL)
```

## Exploit before explore — follow the winners

**Most important experimentation principle.** When a structural change produces a meaningful improvement, *do not wander off to tune hyperparameters*. Ask: **why did this work, and how can I push harder in the same direction?**

Rules:

1. **When a structural change is kept**, your next 2-3 experiments MUST explore variations of that same idea before trying anything unrelated. If more context epochs helped, try even more. If a new conv architecture helped, try variations of it. Exhaust the vein before moving on.

2. **When a structural change crashes (especially OOM)**, that is an engineering problem, not a dead end. Spend at least one experiment trying to make it work — reduce batch size, downsample, use gradient accumulation, etc. The ideas that crash often have the most headroom.

3. **Tag every experiment** in results.tsv as either `(STRUCTURAL)` or `(TUNING)` at the end of the description. Structural = changes what the model sees or how it processes at a fundamental level (input representation, context window, architecture, new augmentation types, loss function). Tuning = adjusting numbers on an existing design (LR, WD, batch size, dropout rate).

4. **Tuning budget**: After exhausting a structural direction, at most **5 consecutive TUNING experiments** before attempting another STRUCTURAL change. If 3 consecutive tuning experiments are discarded, stop tuning immediately and switch to structural.

5. **Track your ratio**: Aim for at least 40% structural experiments.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-sleep/mar27`).

**Track these counters** (in working memory, reset on each run):
- `experiments_since_improvement`: increments on every discard, resets to 0 on every keep
- `consecutive_tuning`: increments on every TUNING experiment, resets to 0 on every STRUCTURAL experiment

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. **Check counters**: If `experiments_since_improvement >= 10`, MUST enter plateau diagnosis. If `consecutive_tuning >= 5`, MUST run a STRUCTURAL experiment next. If last 3 tuning experiments were all discarded, switch to STRUCTURAL immediately.
3. **Think before you code.** What is your hypothesis? Why do you believe this change will help? If you can't articulate a reason grounded in understanding of the data/model/errors, reconsider. "Maybe this will help" is not a hypothesis. If a recent structural change was kept, your hypothesis should build on it.
4. Modify `train.py` (or data loading in `prepare.py`) with an experimental idea.
5. git commit.
6. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
7. Read out results: `grep "^accuracy:\|^kappa:\|^macro_f1:\|^class_.*_f1:\|^peak_vram_mb:" run.log`
8. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix after a few attempts, give up on this idea.
9. Record results in `results.tsv` in the campaign directory. Tag description with `(STRUCTURAL)` or `(TUNING)`. Do NOT commit results.tsv — leave it untracked.
10. If accuracy improved (higher), "advance" the branch, keeping the commit. Reset `experiments_since_improvement = 0`.
11. If accuracy is equal or worse, `git reset` back to where you started. Increment `experiments_since_improvement`.

**Timeout**: Each experiment should take ~10 minutes (+ startup/eval overhead). If a run exceeds 15 minutes, kill it and treat as failure (discard and revert).

**Crashes**: Use judgment. If it's a typo or missing import, fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Do NOT pause to ask the human anything — not during setup, not between experiments, not ever. The human might be asleep or away and expects you to work *indefinitely* until manually stopped. You are fully autonomous. If you run out of ideas, think harder — go back to the data, look at the errors, question assumptions, re-read the components library, re-read the literature in `related-work/`. The loop runs until the human interrupts you, period.
