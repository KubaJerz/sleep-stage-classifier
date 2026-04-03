# Diagnosis — apr03

## Initial Data Exploration

**Actual dataset sizes** (after wake trimming, 58/10/10 split):
- Train: 131,554 epochs | Val: 32,749 epochs

**Class distribution (train / val):**
| Class | Train % | Val % |
|-------|---------|-------|
| Wake  | 27.7%   | 44.8% |
| N1    | 10.4%   | 12.4% |
| N2    | 39.0%   | 27.9% |
| N3    | 8.2%    | 4.9%  |
| REM   | 14.8%   | 10.0% |

Wake trimming reduced Wake from ~69% to ~28% in train. Val still Wake-heavy (44.8%).

**Per-class signal amplitude (train, z-scored):**
| Class | Per-epoch std (mean) |
|-------|---------------------|
| N3    | 1.36 (slow waves)   |
| Wake  | 1.16 (artifacts, eye mvt) |
| N2    | 0.72 (moderate)     |
| N1    | 0.59 (low amplitude) |
| REM   | 0.57 (low amplitude) |

N1 and REM are nearly identical in amplitude — hardest pair to separate with time-domain features alone. Frequency content (spindles in N2, theta in REM, alpha in Wake) is what distinguishes them.

**Temporal structure:**
- Epochs are in subject/temporal order within the concatenated arrays
- 10.4% transition rate — 90% of adjacent epochs share the same label
- Multi-epoch context should provide massive information gain

**Representation analysis — what the current pipeline does vs. could do:**
1. Single-epoch input (3000 samples) — misses temporal context (biggest gap)
2. Raw time-domain only — misses explicit frequency information
3. 2-layer CNN with aggressive downsampling — very limited model capacity
4. No attention — treats all timepoints equally
5. No class weighting — may underfit minority classes

## Current best
*Awaiting baseline*
