# Preprocessing -- What the Model Sees

Preprocessing decides the representation that enters the model. The bias here is toward minimal intervention -- keep the signal close to raw, let the model learn what it needs. The components below cover temporal framing, normalization, and signal conditioning -- the decisions between "raw EEG file" and "tensor fed to the stem."

---

## Epoch Segmentation

**Idea:** Slice continuous polysomnography recordings into fixed-length windows aligned with clinical scoring annotations.

**Why:** Sleep staging is defined per-epoch by clinical convention (AASM: 30 seconds). The epoch is the atomic unit that carries a label. Without segmentation there is no supervised learning problem. The 30-second length is not arbitrary -- it is long enough to contain multiple cycles of the slowest clinically relevant rhythms (delta, ~0.5 Hz) while short enough that sleep stage is approximately stationary within the window.

**Pattern:**
```
continuous_eeg: (total_samples,)
annotation_file -> epoch_boundaries: list of (start_sample, end_sample, label)

for each boundary:
    epoch = continuous_eeg[start : start + fs * epoch_seconds]
    # epoch shape: (fs * epoch_seconds,)
    # e.g. 100 Hz * 30s = (3000,)
```

**Seen in:** All 12 papers. Universal choice of 30s epoch length at the annotation level, though some feed multi-epoch windows to the model (see Context Windowing below).

**Knobs:**
- Sampling rate fs (100 Hz standard for Sleep-EDF; some datasets arrive at 125, 256 Hz and get resampled)
- Epoch length in seconds (30s is standard; 20s used in some MASS subsets)

---

---

## Label Remapping

**Idea:** Map the original scoring vocabulary to a reduced label set suitable for classification.

**Why:** The R&K scoring system uses S1, S2, S3, S4, REM, W -- six classes. Modern AASM collapses S3 and S4 into N3, yielding five classes (W, N1, N2, N3, REM). MOVEMENT and UNKNOWN epochs carry no useful staging information. This remapping is necessary for cross-dataset compatibility and for defining a consistent output space. It is so universal it is easy to forget it is a design decision -- but the merge granularity (e.g., whether to keep S3/S4 separate, or collapse further to 3-class) is a real choice.

**Pattern:**
```
rk_to_aasm = {
    "W": "W",   "S1": "N1",  "S2": "N2",
    "S3": "N3", "S4": "N3",  "REM": "REM",
    "MOVEMENT": EXCLUDE,  "UNKNOWN": EXCLUDE
}
```

**Seen in:** All papers using Sleep-EDF. CSleepNet, CSCNN+HMM, EnsembleNet, ESSN, MicroSleepNet, SCANSleepNet, SCNet, SleepSatelightFTC, SleePyCo, TinySleepNet.

**Knobs:**
- Number of output classes (5 is standard; 3-class W/NREM/REM and 2-class sleep/wake also exist)
- Whether S3 and S4 are merged or kept separate
- What to do with MOVEMENT and UNKNOWN (exclude vs. map to Wake)

---

## Per-Epoch Normalization

**Idea:** Standardize each epoch independently to zero mean and unit variance (z-score), or subtract the mean only (zero-centering).

**Why:** Raw EEG amplitude varies across subjects, sessions, and electrode impedance. Per-epoch normalization removes this offset and scale variation so the model sees shape rather than absolute voltage. Zero-mean-unit-variance is the stronger form; zero-centering alone preserves relative amplitude information within the epoch while removing DC offset. Doing this per-epoch (rather than per-recording or globally) makes each epoch self-contained, which is important when epochs are shuffled or drawn from different recordings.

**Pattern:**
```
# Full z-score (zero mean, unit variance)
epoch_normalized = (epoch - mean(epoch)) / (std(epoch) + eps)

# Zero-centering only
epoch_centered = epoch - mean(epoch)
```

**Seen in:**
- Z-score normalization: CSleepNet
- Zero-centering: EnsembleNet
- No normalization: MicroSleepNet, SCANSleepNet, TinySleepNet, SCNet, SleepSatelightFTC, SleePyCo (majority of papers skip this entirely)

**Knobs:**
- Normalization scope: per-epoch, per-recording, per-channel, or global
- Whether to normalize variance (z-score) or only center (subtract mean)
- Epsilon for numerical stability in division

---

## Bandpass Filtering

**Idea:** Apply a frequency-domain filter to remove signal content outside the physiologically relevant band before the model sees it.

**Why:** EEG useful for sleep staging lives roughly in 0.3--35 Hz. Below 0.3 Hz is electrode drift and DC offset. Above ~40 Hz is mostly muscle artifact and line noise (50/60 Hz). Bandpass filtering removes these irrelevant or harmful components, reducing the burden on the model to learn to ignore them. However, most modern papers skip this -- the argument being that a CNN with learned filters can implicitly do this, and explicit filtering risks removing information the model could have used.

**Pattern:**
```
# FIR bandpass filter
filtered = fir_bandpass(epoch, low=f_low, high=f_high, fs=fs, order=N)
# epoch shape unchanged: (fs * epoch_seconds,)
```

**Seen in:**
- FIR bandpass [0.3, 40] Hz: ESSN
- Implied but unspecified filtering: Portable GUI (Atianashie et al.)
- Explicitly no filtering: MicroSleepNet, TinySleepNet, SCNet, SCANSleepNet, SleePyCo, SleepSatelightFTC, EnsembleNet

**Knobs:**
- Passband (f_low, f_high) -- common choices: [0.3, 35], [0.3, 40], [0.5, 30]
- Filter type (FIR vs. IIR) and order
- Whether to filter at all (most recent papers do not)

---

## Downsampling

**Idea:** Reduce the sampling rate of the signal, typically with an anti-aliasing low-pass filter applied first.

**Why:** Lower sampling rates mean shorter input sequences, which reduces computation (especially for attention-based or recurrent models where cost scales with sequence length). If the useful frequency content is below 25 Hz, sampling at 50 Hz captures everything by Nyquist -- the extra samples at 100 Hz are redundant. The tradeoff is losing high-frequency content above the new Nyquist limit.

**Pattern:**
```
# Anti-alias then decimate
filtered = lowpass(epoch, cutoff=new_fs/2, fs=original_fs)
downsampled = filtered[::decimation_factor]
# (3000,) at 100 Hz -> (1500,) at 50 Hz
```

**Seen in:**
- 100 Hz to 50 Hz with brick-wall anti-alias filter: SleepSatelightFTC
- 125 Hz to 100 Hz: MicroSleepNet (for SHHS dataset), SleePyCo (standardization across datasets)

**Knobs:**
- Target sampling rate (50 Hz, 100 Hz)
- Anti-aliasing filter design (brick-wall, Chebyshev, etc.)
- Whether to resample (interpolation) or decimate (integer factor)


---

## Multi-Epoch Context Windowing

**Idea:** Instead of feeding single 30-second epochs, concatenate or stack multiple consecutive epochs into a wider input window so the model has temporal context around the epoch being classified.

**Why:** Sleep stages do not switch randomly -- they follow structured transition patterns (W->N1->N2->N3 descent, REM cycles every ~90 min). A single 30-second snapshot may be ambiguous (N1 vs. W, N1 vs. REM), but seeing what came before and after resolves the ambiguity. Wider context also lets the model learn transition dynamics directly from the input rather than relying on a separate post-processing step.

**Pattern:**
```
# Concatenation approach (classify center epoch)
context_window = concat(epoch[t-k], ..., epoch[t], ..., epoch[t+k])
# shape: (fs * epoch_seconds * (2k+1),)
# label: stage of epoch[t]

# Sequence approach (classify all epochs in window)
sequence = stack(epoch[t], epoch[t+1], ..., epoch[t+L-1])
# shape: (L, fs * epoch_seconds)
# labels: (L,) -- one per epoch
```

**Seen in:**
- 3 epochs (90s), classify center: SCNet (ACC=86.1%)
- 10 epochs (300s), classify all: SleePyCo (ACC=84.6%)
- 15 epochs (450s), many-to-many: TinySleepNet (ACC=85.4%)
- 25 epochs (~12.5 min), classify center via transfer: SleepSatelightFTC (ACC=85.7%)
- 200 epochs (100 min), classify all: ESSN (ACC=87.1%)
- Full night, classify all: EnsembleNet (ACC=87.07%)
- Single epoch (no context): CSleepNet, CSCNN+HMM, MicroSleepNet, SCANSleepNet

**Knobs:**
- Context length L (number of adjacent epochs)
- Symmetric vs. causal (future context or past only -- causal matters for real-time deployment)
- Concatenation vs. stacking (1D vector vs. sequence of vectors)
- Whether context is handled inside the model (LSTM/Transformer over epoch features) or at the input level (raw concatenation)

---

## Channel Concatenation / Selection

**Idea:** Choose which EEG channels to use, and decide how to combine them if using more than one.

**Why:** Clinical PSG records many channels, but for wearable or edge deployment, fewer channels means fewer sensors. Most sleep staging work uses a single channel (Fpz-Cz is the standard benchmark). When using two channels, they can be stacked as separate input dimensions (multi-channel) or concatenated end-to-end into a longer 1D signal. Concatenation is simpler but destroys the spatial relationship between channels.

**Pattern:**
```
# Single channel (most common)
x = eeg_fpz_cz[start:end]          # shape: (T,)

# End-to-end concatenation
x = concat(eeg_fpz_cz, eeg_pz_oz)  # shape: (2T,)

# Multi-channel stacking
x = stack(eeg_fpz_cz, eeg_pz_oz)   # shape: (2, T)
```

**Seen in:**
- Single Fpz-Cz: CSCNN+HMM, EnsembleNet, ESSN, MicroSleepNet, SCANSleepNet, SCNet, SleepSatelightFTC, SleePyCo, TinySleepNet
- Concatenated Fpz-Cz + Pz-Oz: CSleepNet (6000 samples = 2 x 3000)
- Independent evaluation on Fpz-Cz and Pz-Oz: CSCNN+HMM, SCANSleepNet

**Knobs:**
- Which channel(s) to use (Fpz-Cz, Pz-Oz, C4-A1, F4-M1 -- depends on dataset and hardware)
- Combination method: concatenation, stacking, or independent processing with late fusion
- Whether to include non-EEG channels (EOG, EMG) as additional inputs

---

## Data Augmentation

**Idea:** Apply random transformations to training epochs to increase effective dataset size and reduce overfitting.

**Why:** Sleep datasets are limited in subject count (20--78 for Sleep-EDF). Augmentation synthesizes plausible variations the model should be invariant to -- amplitude differences between subjects, slight temporal misalignment of epoch boundaries, electrode noise. This is especially useful during contrastive pretraining where augmented views of the same epoch serve as positive pairs.

**Pattern:**
```
# Each transform applied independently with probability p
augmented = epoch
if random() < p: augmented *= uniform(1-a, 1+a)        # amplitude scaling
if random() < p: augmented = roll(augmented, shift)     # time shift
if random() < p: augmented += constant                  # DC offset shift
if random() < p: augmented[mask] = 0                    # zero masking
if random() < p: augmented += N(0, sigma)               # Gaussian noise
if random() < p: augmented = bandstop(augmented, f1,f2) # band-stop filter
```

**Seen in:**
- 6 sequential transforms (p=0.5 each): SleePyCo (amplitude scaling, time shift, amplitude shift, zero-masking, Gaussian noise, band-stop filter -- used during contrastive pretraining)
- Signal shift +/-10%: TinySleepNet
- Sequence offset 0--5 epochs: TinySleepNet
- No augmentation: MicroSleepNet (explicitly stated), most other papers

**Knobs:**
- Which transforms to include and their probability p
- Amplitude scaling range
- Time shift range (samples or percentage of epoch)
- Noise variance
- Whether augmentation is applied during all training or only pretraining
- Whether augmentations are composed sequentially or applied individually

---

## No Preprocessing (Raw End-to-End)

**Idea:** Feed the raw digitized EEG signal directly to the model with no filtering, normalization, or spectral transform.

**Why:** The argument is that any hand-designed preprocessing may discard information useful for classification. A sufficiently powerful model can learn to normalize, filter, and decompose the signal on its own through its early layers. This also simplifies the pipeline and removes preprocessing hyperparameters. The practical observation is that several competitive models achieve strong results on raw input, suggesting that for single-channel EEG at 100 Hz, explicit preprocessing is not necessary.

**Pattern:**
```
x = raw_eeg_epoch   # shape: (T,)
# No filtering, no normalization, no spectral transform
# Directly to model stem
```

**Seen in:** MicroSleepNet (ACC=82.8%, explicitly "no preprocessing techniques, data augmentation or class balancing"), SCNet (ACC=86.1%, "raw single-channel EEG end-to-end"), SCANSleepNet (ACC=85.52%), TinySleepNet (ACC=85.4%, "no filtering applied"), SleePyCo (ACC=84.6%, no signal processing beyond resampling).

**Knobs:**
- This is the absence of preprocessing -- but the model's stem design implicitly takes over the job (large first-layer kernels often act as learned bandpass filters)

