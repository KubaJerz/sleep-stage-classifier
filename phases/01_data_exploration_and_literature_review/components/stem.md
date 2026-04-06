# Stem

First contact with the raw signal (or its spectrogram). The stem decides how the model initially downsamples the input and what kind of low-level features it extracts. Everything downstream sees the stem's output, so choices here set the resolution, channel width, and temporal granularity for the rest of the network.

Key tensions: aggressive downsampling saves compute but throws away fine structure. Wide kernels see more context per step but blur fast transients. Parallel branches capture multiple scales but cost more parameters and need alignment before concatenation.

---

## Single Large-Kernel Conv

**Idea:** One wide 1D convolution with a large stride does the initial feature extraction and downsampling in a single shot.

**Why:** A half-second or wider kernel integrates enough signal to capture the dominant waveform shapes in each sleep stage (spindles, K-complexes, slow waves) right at the front door. The large stride aggressively compresses the temporal axis so the rest of the network operates on a short, dense feature sequence. Simple, cheap, and surprisingly effective -- the first conv does most of the spatial reduction so later layers can focus on refinement rather than downsampling.

**Pattern:**
```
Input: (B, 1, T)

Conv1D(C_out, k=K, s=S) + BN + ReLU
  -> (B, C_out, T//S)

Pool(p, s_p)
  -> (B, C_out, T//(S*s_p))
```
Typical values: K ~ 0.5s worth of samples (e.g. 50 at 100Hz), S ~ K//2.
C_out in the range 64-128.

Some variants double the channels at pooling by concatenating MaxPool and AvgPool outputs side by side (M-APooling), capturing both peak activations and average energy.

**Seen in:** TinySleepNet (k=50, s=25 at 100Hz; ACC 85.4% EDF-20), SingleChannelNet (k=128, s=2 with M-APooling; ACC 86.1% EDF-78).

**Knobs:** Kernel width relative to sampling rate (how many milliseconds of signal the first conv sees), stride (how much temporal compression happens immediately), whether to use M-APooling vs plain MaxPool (doubles channel width for free diversity).

---

## Dual-Scale Parallel Branches

**Idea:** Two parallel 1D conv branches with different kernel sizes -- one small, one large -- process the same input and get concatenated.

**Why:** EEG sleep stages are defined by features at different timescales. Delta waves (N3) are slow and wide; spindles and alpha bursts are fast and narrow. A single kernel size is a compromise. Two branches let the network see both scales natively without forcing one kernel to do double duty. The small kernel (~0.5s) catches fast transients, the large kernel (~4s) catches slow oscillations.

**Pattern:**
```
Input: (B, 1, T)

Branch A (fast features):
  Conv1D(C, k=K_small, s=S_small) + ReLU -> MaxPool(p_a, s_a)
  -> (B, C, T_a)

Branch B (slow features):
  Conv1D(C, k=K_large, s=S_large) + ReLU -> MaxPool(p_b, s_b)
  -> (B, C, T_b)

Concat along time dim -> (B, C, T_a + T_b)
```
Choose strides and pool sizes so T_a and T_b are in a reasonable ratio (not necessarily equal -- concatenating along time is fine if downstream layers are translation-invariant).

Typical: K_small ~ 25-50 samples, K_large ~ 100-400 samples at 100Hz.

**Seen in:** CSCNN/Improved-SENet (k=50 + k=400 at 100Hz; ACC 84.6% EDF-20), SCANSleepNet (k=25 + k=100; ACC 85.52% EDF-20).

**Knobs:** Ratio between the two kernel sizes (how far apart the scales are), whether branches share the same number of filters, how to align temporal dimensions before concatenation (stride/pool tuning vs padding), activation choice (ReLU vs Mish).

---

## Multi-Scale Conv Bank

**Idea:** Many parallel convolutions (5-10+) with a wide spread of kernel sizes, all applied to the same input, concatenated depth-wise.

**Why:** Instead of picking two scales, cast a wide net. This is the Inception intuition applied to raw EEG: let the network have access to everything from sub-second to multi-second patterns and learn which matter. Each filter is lightweight (sometimes just 1 output channel each), so the total cost stays reasonable despite the fan-out.

**Pattern:**
```
Input: (B, 1, T)

N parallel Conv1D branches:
  Conv1D(c_i, k=K_i, s=1, same_pad) for K_i in {K_1, K_2, ..., K_N}
  -> each produces (B, c_i, T)

DepthConcat -> (B, sum(c_i), T)
BN + ReLU + MaxPool(p, s_p) -> (B, sum(c_i), T//s_p)
```
Kernel range might span K_1=51 to K_N=501 (0.5s to 5s at 100Hz).
Individual c_i can be as small as 1 (one filter per kernel size).

Can be repeated as a second, narrower bank on the downsampled output.

**Seen in:** EnsembleNet (10 parallel kernels 51-501 in first bank, 5 kernels in second bank; ACC 87.07% EDF-20).

**Knobs:** Number of parallel branches, kernel size range and spacing (linear vs logarithmic), filters per branch (uniform vs scaled), number of stacked banks, pooling factor between banks.

---

## Deep Sequential Conv Stack

**Idea:** A chain of conv-pool-norm blocks that progressively downsample and widen channels, standard deep CNN style.

**Why:** Simple and well-understood. Each layer extracts slightly more abstract features from the previous layer's output. Works well when you have enough data to train the depth. The gradual downsampling preserves information better than one aggressive step, at the cost of more layers and parameters.

**Pattern:**
```
Input: (B, 1, T)

for i in 1..N_blocks:
  Conv1D(C_i, k=K, s=S) + ReLU
  MaxPool(p, s_p)
  BatchNorm (optional)

-> (B, C_N, T_final)
```
Channel progression typically widens: 64 -> 128 -> 256 or similar.
Kernel sizes may decrease as the signal gets shorter (k=7 early, k=5 later).
3-7 conv layers is common. Dropout at the end before handing off.

**Seen in:** CSleepNet (7 conv layers, 64->128->256->128; ACC 86.41% EDFX), Raspberry Pi CNN (3 conv blocks, k=125->50->10; deployed on RPi 3B+, 82K params).

**Knobs:** Depth (number of conv layers), channel width progression, kernel sizes (uniform vs decreasing), where to place batch norm (after every layer vs every few), dropout placement and rate, stride vs pool for downsampling.

---

## Lightweight Group Conv with Attention

**Idea:** Group convolutions (each filter group only sees a subset of input channels) with channel shuffle and per-block attention, designed for extreme parameter efficiency.

**Why:** Standard convolutions at 128 channels are expensive in multiply-adds. Group convolutions cut that cost by a factor of G (the number of groups) but isolate channel groups from each other. Channel shuffle after each block restores cross-group communication. Adding a lightweight attention module (channel + spatial) after each block lets the network dynamically emphasize the features that matter for the current input, compensating for the reduced per-group capacity.

**Pattern:**
```
Input: (B, 1, T)

Block 1: GConv(C, k=K, groups=1) + BN + LeakyReLU + MaxPool(3,3) + Attn -> (B, C, T//3)
Block 2: GConv(2C, k=K, groups=G) + BN + LeakyReLU + MaxPool(3,3) + Attn -> (B, 2C, T//9)
  ChannelShuffle
Block 3..N: repeat with groups=G, same 2C width
  ChannelShuffle between blocks

-> (B, 2C, T_final)
```
Attention can be ECA-style (1D conv on channel descriptor) + spatial attention (max/mean pool across channels -> conv -> sigmoid gate).

**Seen in:** MicroSleepNet (48K params total, 5 blocks, group conv + ECSA attention + channel shuffle; ACC 82.8% EDF-20, 2.8ms inference on mobile).

**Knobs:** Number of groups G (tradeoff: more groups = cheaper but more isolated), attention type (channel-only vs channel+spatial), number of blocks before handing off, whether first block uses groups=1 (full conv) to mix the raw input channels.

---

## Progressive Small-Kernel CNN with SE

**Idea:** Many small-kernel (k=3) convolutions grouped into blocks, with squeeze-and-excite attention per block and aggressive pooling between blocks.

**Why:** Small kernels are cheap and composable -- three stacked k=3 convs have the same receptive field as one k=7 but with more nonlinearities and fewer parameters. SE modules add a learned channel-weighting step that's nearly free in compute. Large pool strides between blocks (pool=5) do the heavy temporal compression. This gives a VGG/ResNet-flavored stem where depth and pooling do the work, not wide kernels.

**Pattern:**
```
Input: (B, 1, T)

Block_i (repeated R times inside each block):
  Conv1D(C_i, k=3, pad=same) + BN + PReLU
  (last repeat includes SE module)
MaxPool(P)

Stack N blocks:
  Block 1: C_1 filters, pool P -> (B, C_1, T//P)
  Block 2: C_2 filters, pool P -> (B, C_2, T//P^2)
  ...
  Block N: C_N filters, no pool -> (B, C_N, T//P^(N-1))
```
SE module: GAP -> FC(C//r) -> ReLU -> FC(C) -> Sigmoid -> channel-wise scale.
Typical: R=3 convs per block, P=5, N=5 blocks, C progression 64->128->192->256->256.

**Seen in:** SleePyCo (5 blocks, 3 convs each, SE in last conv of each block, pool=5; ACC 84.6% EDF with contrastive pretraining).

**Knobs:** Convolutions per block R, pool stride P (controls total compression), SE reduction ratio r, channel width progression, activation (PReLU vs ReLU vs GELU), whether the last block pools or not (affects output sequence length).

---

## Parallel Domain Branches

**Idea:** Separate conv stems independently process two representations of the same epoch -- typically raw time-domain signal and its frequency-domain transform -- then merge downstream.

**Why:** Time and frequency representations emphasize different things. Raw EEG preserves waveform morphology and transient timing. A spectral transform (amplitude spectrum, multi-taper) makes frequency band power explicit and is more robust to phase shifts. Processing them with independent stems lets each branch specialize, and the merge point gets both perspectives.

**Pattern:**
```
Time branch:
  Input: (B, T_time)
  Conv1D(C, k=K_t) + optional DWConv2D + BN + ReLU + Dropout + AvgPool
  -> (B, C', T_time')

Freq branch:
  Input: (B, T_freq)
  Conv1D(C, k=K_f) + optional DWConv2D + BN + ReLU + Dropout + AvgPool
  -> (B, C', T_freq')

[each branch continues through independent backbone blocks]
[flatten and concat before head]
```
The frequency input can be a log-amplitude spectrum, multi-taper spectral estimate, or STFT magnitude. Branches may share architecture but not weights, or they can differ in depth/width.

**Seen in:** SleepSatelightFTC (Conv1D + depthwise Conv2D per branch, time at 1500 samples + freq at 750 samples, 50Hz; ACC 85.7% EDF-20).

**Knobs:** What frequency representation to use (amplitude spectrum, STFT, wavelet), whether branches share architecture, where to merge (early concat vs late concat vs attention-weighted fusion), relative capacity allocated to each branch.

---

## Linear Projection from Spectrogram

**Idea:** Skip convolutional feature extraction entirely. Start from a pre-computed time-frequency representation (STFT, mel spectrogram) and use a linear layer to compress the frequency axis.

**Why:** If you've already transformed the signal into a spectrogram, each time frame is a frequency vector -- and a linear projection is the cheapest possible dimensionality reduction. This works when the downstream model (e.g., BiLSTM, Transformer) is strong enough to learn temporal patterns from the projected features. It also means the "stem" is essentially free, pushing all capacity into the sequence model.

**Pattern:**
```
Input: (B, T_frames, F_bins)     e.g. from STFT

Linear(F_bins -> H) + SiLU
-> (B, T_frames, H)
```
T_frames comes from the STFT windowing (e.g., 29 frames for 30s with 2s window, 1s hop).
F_bins depends on FFT size (e.g., 128 bins from 256-point FFT).
H is a small hidden dim (32-64).

This is not a "no-stem" -- the STFT *is* the feature extraction, and the linear layer is the dimensionality bottleneck.

**Seen in:** ESSN (STFT with 2s Hamming window, 1s overlap, 256 FFT -> 29x128 per epoch, linear 128->32 + SiLU; ACC 87.1% EDF-20, 0.27M params).

**Knobs:** STFT parameters (window length, hop, FFT size -- these control the time-frequency tradeoff), log transform vs raw magnitude, projection dimension H, activation after projection, whether to add a recurrent layer immediately after (as ESSN does with BiLSTM).
