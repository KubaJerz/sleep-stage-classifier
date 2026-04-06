# Component Template

Each component file covers one **place** in the pipeline. Within each file, individual components are listed as H2 sections.

---

## Component Name

**Idea:** One sentence — what this does.

**Why:** The intuition. What problem does this solve, what property of the signal or task makes this work. Not a justification — a mental model the researcher can carry forward.

**Pattern:**
```
Concise structural sketch showing the flow.
Use parameterized dimensions (C, T, H, etc.) not hardcoded numbers.
Show shapes at key points so the researcher can sanity-check.
Keep it minimal — just enough to implement from, not a spec sheet.
```

**Seen in:** Paper names that used this (with key metric if relevant).

**Knobs:** What a researcher would tune when adapting this — the meaningful degrees of freedom. Not an exhaustive hyperparameter list, just the choices that actually change behavior.

---

## Places

| Place | File | What it decides |
|-------|------|-----------------|
| Preprocessing | `preprocessing.md` | What the model sees |
| Stem | `stem.md` | First contact with the signal |
| Feature Processing | `feature_processing.md` | Refinement, attention, temporal modeling — any order, any combination |
| Head | `head.md` | Collapse features to class probabilities |
| Loss | `loss.md` | What the model optimizes for |
| Optimizer/Training | `optimizer.md` | How the model learns |
