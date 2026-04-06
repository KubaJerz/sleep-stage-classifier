# Feature Processing

Composable building blocks for refining, recombining, and sequencing features after initial extraction. These are not ordered stages -- they are independent ideas that can be wired in wherever they make sense. A model might use one, several, or none. They can precede each other, follow each other, or wrap around each other. The only constraint is that tensor shapes must agree at the boundaries.

---

## Channel Attention (Squeeze-and-Excitation)

**Idea:** Learn a per-channel importance weight by squeezing the spatial dimension, then re-scale each channel accordingly.

**Why:** Not all feature channels are equally useful for every input. A spindle-heavy epoch should amplify the channel that captured spindle energy; a delta-heavy epoch should amplify different channels. Global average pooling collapses spatial variation into a single descriptor per channel, and the two FC layers learn a non-linear mapping from that descriptor to a gating weight. This is cheap -- it adds very few parameters -- and it can be inserted after any conv layer.

**Pattern:**
```
Input: (B, C, T)

GAP over T:          (B, C, T) -> (B, C)
FC(C -> C/r) + ReLU: (B, C/r)         # r is the reduction ratio
FC(C/r -> C) + Sigmoid: (B, C)        # channel weights, each in [0,1]
Scale:               input * weights.unsqueeze(-1) -> (B, C, T)
```

**Seen in:** SleePyCo (SE modules in each conv block; ACC 84.6%), CSCNN/Improved-SENet (channel branch of CSAM; ACC 84.6%).

**Knobs:** Reduction ratio r (typically 2 or 4 -- smaller r = more expressive but more params). Where to insert it (after every block, or only the last). Whether to use GAP, GMP, or both as the squeeze operation.

---

## Spatial Attention

**Idea:** Learn a per-timestep importance weight by compressing across channels, then re-scale each timestep.

**Why:** Within a 30-second epoch, not every time window matters equally. A K-complex occupies maybe 1-2 seconds; the rest is background. Spatial attention lets the model suppress uninformative timesteps and amplify the ones that carry discriminative structure. It is the transpose of channel attention -- one weights channels, the other weights positions.

**Pattern:**
```
Input: (B, C, T)

Pool over C (mean and/or max): (B, 1, T) or (B, 2, T)
Conv1D(k=k_s) + Sigmoid:      (B, 1, T)       # spatial weights
Scale: input * weights -> (B, C, T)
```

**Seen in:** CSCNN/Improved-SENet (spatial branch of CSAM, fused additively with channel attention; ACC 84.6%), MicroSleepNet (spatial part of ECSA; ACC 82.8%).

**Knobs:** How to compress channels (mean, max, or concat both). Kernel size of the spatial conv (larger = more context for the gating decision). Whether to combine with channel attention (additive, multiplicative, sequential).

---

## Combined Channel-Spatial Attention (CSAM / ECSA)

**Idea:** Apply both channel and spatial attention, then fuse the two views.

**Why:** Channel attention knows which features matter; spatial attention knows where they matter. Combining them gives the model a two-axis recalibration that neither alone provides. The fusion strategy matters -- additive fusion lets both signals contribute independently, while sequential application (channel then spatial) conditions the spatial decision on the channel-recalibrated features.

**Pattern:**
```
Input: (B, C, T)

Channel path:
  GAP over T -> FC(C->C/r) + ReLU -> FC(C/r->C) + Sigmoid -> Ac: (B, C, 1)

Spatial path:
  Pool over C -> FC or Conv1D -> Sigmoid -> As: (B, 1, T)

Fusion (additive):  output = input * Ac + input * As
   -- or --
Fusion (sequential): output = (input * Ac) * As
```

**Seen in:** CSCNN/Improved-SENet (additive CSAM; +1.55% ACC over no attention), MicroSleepNet (ECSA with ECA-style channel attention + spatial attention; 48K params total model).

**Knobs:** Fusion strategy (additive vs. sequential vs. parallel-then-concat). Whether channel attention uses FC bottleneck (SE-style) or 1D conv (ECA-style, no FC). Reduction ratio. Number of insertion points in the network.

---

## Dilated (Atrous) Convolutions

**Idea:** Use convolutions with gaps between kernel elements to expand the receptive field without adding parameters or reducing resolution.

**Why:** After the stem has downsampled the signal, you often have a compact feature map (say T=37 timesteps) but need to capture patterns that span many of those timesteps. Standard convolutions with small kernels see only local neighborhoods. Stacking dilated convolutions with exponentially increasing dilation rates (1, 2, 4, ...) gives each layer a progressively wider view while keeping the kernel size small. The receptive field grows exponentially with depth instead of linearly.

**Pattern:**
```
Input: (B, C_in, T)

DilConv(k=K, d=1,  C_out=C1) + BN + Act: (B, C1, T)    # receptive field = K
DilConv(k=K, d=2,  C_out=C2) + BN + Act: (B, C2, T)    # receptive field = K + (K-1)*1
DilConv(k=K, d=D3, C_out=C3) + BN + Act: (B, C3, T)    # keeps growing

# Padding chosen to preserve T at each layer
# Total receptive field with L layers, dilation doubling: K * (2^L - 1) / (2 - 1)
```

**Seen in:** MicroSleepNet (3 layers, d=1,2,4, k=3, receptive field of 15; ACC 82.8%).

**Knobs:** Number of layers and dilation schedule (exponential doubling is standard but not mandatory). Kernel size K. Whether to use causal padding (for streaming) or symmetric padding. Whether to add residual connections across the stack.

---

## Multi-Scale Atrous Convolution (MSAC)

**Idea:** Run parallel dilated convolutions at different rates on the same input, then concatenate.

**Why:** Different sleep features live at different temporal scales simultaneously. Delta waves are slow (~0.5-4 Hz), spindles are faster (~11-16 Hz), and K-complexes are transient. Rather than processing these sequentially through stacked dilated layers, MSAC captures all scales in one shot by branching. This is Inception-style thinking applied to dilation rates instead of kernel sizes.

**Pattern:**
```
Input: (B, C, T)

Branch 1: Conv(k=1, d=1, C_out=C') -> (B, C', T)
Branch 2: Conv(k=1) -> DilConv(k=K, d=d1, C') -> (B, C', T)
Branch 3: Conv(k=1) -> DilConv(k=K, d=d2, C') -> (B, C', T)
Branch 4: Conv(k=1) -> DilConv(k=K, d=d3, C') -> (B, C', T)

Concat: (B, 4*C', T) -> BN + Act
```

**Seen in:** SCANSleepNet (MSAC module with hierarchically stacked dilated convs at rates 1, 3, 5; 0.26M params, ACC 85.52%).

**Knobs:** Number of parallel branches. Dilation rates per branch. Whether branches are purely parallel or hierarchically stacked (each branch builds on the previous). The 1x1 conv bottleneck width.

---

## Inception-Style Multi-Scale Conv Blocks

**Idea:** Run parallel convolution branches with different kernel sizes on the same input, then concatenate along the channel dimension.

**Why:** The optimal kernel size depends on the feature you are trying to detect, and you do not know in advance which features matter for a given epoch. Small kernels (k=1,3) capture sharp transients; medium kernels (k=16) capture waveform morphology; large kernels (k=64) capture slow oscillation envelopes. Running them all in parallel and concatenating lets the next layer decide which scale matters.

**Pattern:**
```
Input: (B, C_in, T)

Branch 1: Conv(k=1, C_out=C1)                      -> (B, C1, T)
Branch 2: Conv(k=1, C_out=C') -> Conv(k=K_med, C2)  -> (B, C2, T)
Branch 3: Conv(k=1, C_out=C') -> Conv(k=K_lg, C3)   -> (B, C3, T)
Branch 4: Pool(k=3) -> Conv(k=1, C_out=C4)          -> (B, C4, T)

Concat: (B, C1+C2+C3+C4, T)
```

**Seen in:** SCNet (MC blocks with k=1,3,16,64 + M-APooling branch, 272 output channels; ACC 86.1%), EnsembleNet (multi-scale parallel convs with k=51 to k=501; ACC 87.07%).

**Knobs:** Number of branches and their kernel sizes. Whether to use 1x1 bottlenecks before expensive convolutions. Filter allocation across branches (more filters on which scales). How many MC blocks to stack.

---

## Unidirectional LSTM

**Idea:** Process a sequence of feature vectors with a single-direction LSTM, using the hidden states as contextualized representations.

**Why:** Sleep stages are not independent -- a transition from N2 to REM is meaningful, a jump from N3 to W is suspicious. An LSTM carries forward a compressed summary of everything it has seen so far, letting each timestep's representation be informed by its history. Unidirectional is sufficient when the model only needs past context (e.g., causal inference, streaming), or when the sequence is short enough that backward context adds little.

**Pattern:**
```
Input: (B, L, D)         # L timesteps, D features per step

LSTM(hidden=H):
  -> all states: (B, L, H)    # use this for many-to-many (one prediction per step)
  -> final state: (B, H)      # use this for many-to-one (single prediction)

Dropout(p)
```

**Seen in:** CSleepNet (single LSTM, H=1024, operates on intra-epoch feature map timesteps; ACC 86.41%), TinySleepNet (LSTM H=128, operates across 15-epoch sequences, many-to-many; ACC 85.4%).

**Knobs:** Hidden size H. Number of stacked LSTM layers. Whether to use the final state only (many-to-one) or all states (many-to-many). What constitutes a "timestep" -- could be time steps within an epoch's feature map, or entire epochs in a sequence. Dropout between layers.

---

## Bidirectional LSTM

**Idea:** Process a sequence in both directions and concatenate forward/backward hidden states.

**Why:** In offline sleep staging (where the entire night is available), there is no reason to restrict the model to past context only. A BiLSTM lets each timestep attend to both its history and its future, which is especially useful for transitions -- knowing that N2 follows can help disambiguate an ambiguous N1 segment. The cost is doubled parameters and no streaming capability.

**Pattern:**
```
Input: (B, L, D)

BiLSTM(hidden=H):
  -> (B, L, 2H)    # forward H + backward H concatenated at each step

Optional: stack multiple BiLSTM layers
  BiLSTM_1(H) -> Dropout -> BiLSTM_2(H) -> (B, L, 2H)
```

**Seen in:** EnsembleNet (BiLSTM H=128, full-night sequences of 90-dim vectors, 10 independently trained for ensemble; ACC 87.07%), ESSN (stacked BiLSTM in both epoch encoder and sequence encoder; ACC 87.1%).

**Knobs:** Hidden size H. Number of stacked layers. Whether to use residual connections between layers. What level the sequence operates at (intra-epoch frames, or inter-epoch full-night). Ensemble size if training multiple independently.

---

## Transformer Encoder

**Idea:** Apply multi-head self-attention followed by feedforward layers to let every position attend to every other position.

**Why:** Self-attention has no inherent locality bias -- it can relate any two timesteps regardless of distance, with the relationship strength learned from data. For sleep staging, this means a W epoch at position 3 can directly inform classification of an ambiguous epoch at position 200 if the model learns that pattern. The multi-head mechanism lets different heads specialize in different kinds of relationships (e.g., one head for local transitions, another for long-range sleep cycle structure).

**Pattern:**
```
Input: (B, L, D)

+ Positional Encoding (sinusoidal or learned): (B, L, D)

Repeat N_layers times:
  MultiHeadAttn(Q=K=V=X, heads=H_attn, d_k=D/H_attn) + Residual + LayerNorm
  FFN(D -> D_ff -> D) + Residual + LayerNorm

Output: (B, L, D)
```

**Seen in:** SleePyCo (6-layer, 8-head transformer with shared weights across 3 feature pyramid levels; ACC 84.6%).

**Knobs:** Number of layers N_layers. Number of attention heads H_attn. Feedforward hidden dimension D_ff. Positional encoding type. Whether weights are shared across different inputs (as in SleePyCo's pyramid). Dropout in attention and FFN.

---

## Position Attention (Learned Query Attention)

**Idea:** Use a small set of learned query vectors to attend over a temporal sequence, producing a fixed-size output that summarizes "what matters where."

**Why:** When you have a variable-length sequence of features and need a fixed-size summary, you need some kind of pooling. GAP treats all positions equally. Max pooling keeps only the loudest signal. Learned query attention lets the model learn what to look for -- each query vector acts as a "question" that gets answered by attending over the sequence. With Q queries, you get Q answer vectors, which can then be summed or concatenated.

**Pattern:**
```
Input: (B, T, H)

Queries: (Q, H) -- learned parameters, Q << T
Keys:    (B, T, H) -- from input

Scores = Queries @ Keys^T:        (B, Q, T)
Optional: Conv1D(k=k_s) over T dimension of scores, then Sigmoid
Weights = Softmax or Sigmoid(Scores): (B, Q, T)

Output = Weights @ Input:          (B, Q, H)
Collapse: sum over Q ->           (B, H)
```

**Seen in:** ESSN/SPAM (Q=4 learned queries, Conv1D smoothing of attention scores, sigmoid gating, sum collapse; ACC 87.1%), SleePyCo (attention pooling to collapse transformer output to single vector).

**Knobs:** Number of query vectors Q. Whether to smooth attention scores (Conv1D kernel size). Softmax vs. sigmoid normalization (softmax forces competition between positions, sigmoid allows multiple positions to be selected independently). Whether to sum or concatenate the Q outputs.

---

## Slice-and-Transpose Sequence Processing

**Idea:** Split a long sequence into fixed-size slices, process each slice independently, then transpose and re-process so that information flows across slices.

**Why:** BiLSTMs on very long sequences (200+ epochs = a full night) are slow and struggle with gradient flow. Slicing into groups of k makes each BiLSTM call manageable. But slicing alone means slices cannot communicate. The transpose step re-groups the data so that position i from every slice is now in the same sequence -- a second BiLSTM pass on the transposed view lets cross-slice information flow. Two short BiLSTMs with a transpose between them approximate one very long BiLSTM at a fraction of the cost.

**Pattern:**
```
Input: (B, N, D)         # N epochs, D features each

Slice into groups of k:  (B, N/k, k, D)

Pass 1 -- within slices:
  Per slice: Linear(D->H) + Act + BiLSTM(H) -> (k, 2H)
  + Residual + Dropout
  Result: (B, N/k, k, 2H)

Transpose: (B, k, N/k, 2H)

Pass 2 -- across slices:
  Per group: Linear(2H->H) + Act + BiLSTM(H) -> (N/k, 2H)
  + Residual + Dropout
  Result: (B, k, N/k, 2H)

Reshape: (B, N, 2H)
```

**Seen in:** ESSN (k=10, N=200, two ESM passes with SPAM attention at each; 0.27M params, ACC 87.1%).

**Knobs:** Slice size k (controls the tradeoff between intra-slice context and number of slices). Whether to use attention (SPAM) within each pass. Number of transpose rounds. Hidden size of each BiLSTM.

---

## Feature Pyramid with Lateral Connections

**Idea:** Tap intermediate layers of a deep backbone at multiple resolutions and project them to a common channel dimension.

**Why:** Early layers have high temporal resolution but shallow features; deep layers have rich features but coarse resolution. A feature pyramid gives the downstream classifier access to all levels simultaneously, so it does not have to rely solely on the deepest (most compressed) representation. Lateral 1x1 convolutions align channel dimensions across levels.

**Pattern:**
```
Backbone outputs at layers i, j, k:
  C_i: (B, C_i, T_i)   # high resolution, shallow features
  C_j: (B, C_j, T_j)   # medium
  C_k: (B, C_k, T_k)   # low resolution, deep features

Lateral projections:
  F_i = Conv1x1(C_i -> C_common): (B, C_common, T_i)
  F_j = Conv1x1(C_j -> C_common): (B, C_common, T_j)
  F_k = Conv1x1(C_k -> C_common): (B, C_common, T_k)

Each F_* is processed independently or jointly by downstream blocks.
```

**Seen in:** SleePyCo (3-level pyramid from Conv3/4/5, C_common=128, each level fed to shared transformer + attention pool, logits summed; ACC 84.6%).

**Knobs:** Which backbone layers to tap (and how many levels). Common channel dimension C_common. Whether to add top-down fusion (FPN-style) or keep levels independent. How to combine level-wise predictions (sum logits, concatenate features, weighted average).

---

## Causal Convolution

**Idea:** Convolutions that can only see the current and past timesteps, never the future.

**Why:** If the model must work in a streaming or real-time setting, it cannot peek ahead. Causal convolutions enforce this by padding only on the left side. Stacking causal convolutions (especially with dilation) builds a temporal receptive field that grows into the past. This is also useful as a regularizer -- restricting information flow can sometimes improve generalization when the task has a natural temporal direction.

**Pattern:**
```
Input: (B, C, T)

CausalConv1D(k=K, d=d):
  left_pad = (K - 1) * d
  Pad input on left only: (B, C, T + left_pad)
  Conv1D(k=K, d=d): (B, C_out, T)     # output length = input length

Stack multiple with increasing dilation for large receptive field.
```

**Seen in:** SCANSleepNet (causal convolution in TFC block, paired with SCDA attention; ACC 85.52%).

**Knobs:** Kernel size K. Dilation schedule. Number of stacked layers. Whether to pair with attention after each layer or only at the end. Residual connections.

---

## Channel Shuffle

**Idea:** After group convolution, rearrange channels so that each group in the next layer receives channels from all previous groups.

**Why:** Group convolution is parameter-efficient (each group processes only C/g channels), but it creates information silos -- group 1 never sees group 2's features. Channel shuffle breaks these silos by interleaving channels across groups. It is a zero-parameter, zero-FLOP operation (just a reshape + transpose + reshape) that restores cross-group information flow.

**Pattern:**
```
Input: (B, C, T) with g groups

Reshape:   (B, g, C/g, T)
Transpose: (B, C/g, g, T)
Reshape:   (B, C, T)

# Channels from different groups are now interleaved
```

**Seen in:** MicroSleepNet (channel shuffle after group conv blocks 2-4; 48K params, mobile deployment at 2.8ms on Snapdragon 865).

**Knobs:** Applied after every group conv or only periodically. Number of groups g (must match the group conv it follows). Can be replaced by a 1x1 conv for a learned (but more expensive) version.

---

## Contrastive Representation Learning (Pretraining)

**Idea:** Before training the classifier, pretrain the feature extractor so that embeddings of same-class epochs are pulled together and different-class epochs are pushed apart.

**Why:** Cross-entropy on a randomly initialized network must simultaneously learn good representations and good decision boundaries. Contrastive pretraining decouples these: first learn a feature space where sleep stages form tight, separated clusters, then train a simple classifier on top. This is especially valuable when labeled data is limited or class boundaries are fuzzy (N1 vs. N2, N1 vs. W). The pretrained representations transfer better across datasets.

**Pattern:**
```
Pretraining (CRL step):
  Backbone: epoch -> (B, D_feat) via GAP
  Projection MLP: D_feat -> D_hidden -> D_z
  Supervised Contrastive Loss(temperature=tau):
    pull together embeddings with same sleep stage label
    push apart embeddings with different labels

Fine-tuning (classification step):
  Remove projection MLP
  Freeze or unfreeze backbone
  Attach classifier head: D_feat -> 5 classes
```

**Seen in:** SleePyCo (supervised contrastive pretraining, tau=0.07, projection d_z=128, then MTCL fine-tuning with frozen backbone; ACC 84.6%).

**Knobs:** Temperature tau (lower = harder negatives). Projection head architecture and dimension d_z. Batch size (larger = more negatives per anchor). Whether to freeze backbone during fine-tuning or allow gradual unfreezing. Augmentation strategy during pretraining (amplitude scaling, time shift, noise, band-stop).

---

## Dual-Pooling (Max + Average Concatenation)

**Idea:** Apply both max pooling and average pooling at the same spatial positions, then concatenate the results along the channel dimension.

**Why:** Max pooling preserves the strongest activation (good for detecting sharp events like K-complexes), while average pooling preserves the overall energy level (good for sustained rhythms like delta waves). Concatenating both gives the next layer access to both views, at the cost of doubling the channel dimension. This is a zero-parameter alternative to learning a pooling strategy.

**Pattern:**
```
Input: (B, C, T)

MaxPool1D(k, s):  (B, C, T')
AvgPool1D(k, s):  (B, C, T')

Concat on C:      (B, 2C, T')
```

**Seen in:** SCNet/M-APooling (used as the primary pooling operation throughout the backbone, doubling channels at each stage; ACC 86.1%).

**Knobs:** Where to apply it (every pooling layer or only at select points). Pool kernel size and stride. Whether to follow with a 1x1 conv to compress back to C channels.

---

## Group Convolution

**Idea:** Split input channels into g groups and apply independent convolutions to each group, reducing parameters by a factor of g.

**Why:** Standard convolution connects every input channel to every output channel. When the channel count is high, this is expensive. Group convolution partitions channels into independent groups, each with its own filters. This is a direct parameter and compute reduction with minimal accuracy cost, especially valuable for edge deployment where every multiply counts. When paired with channel shuffle, cross-group information flow is restored.

**Pattern:**
```
Input: (B, C_in, T)

GroupConv1D(C_in -> C_out, k=K, groups=g):
  Each group: (B, C_in/g, T) -> Conv1D(C_out/g, k=K) -> (B, C_out/g, T)
  Concat groups: (B, C_out, T)

Parameters: K * (C_in/g) * (C_out/g) * g = K * C_in * C_out / g
  (g times fewer than standard conv)
```

**Seen in:** MicroSleepNet (g=64 for blocks 2-5, total model 48K params, ~100KB on mobile; ACC 82.8%).

**Knobs:** Number of groups g (extreme: g=C_in gives depthwise convolution). Whether to follow with channel shuffle or 1x1 pointwise conv. Group count can vary by layer.

---

## Depthwise Separable Convolution

**Idea:** Factor a standard convolution into a depthwise conv (one filter per input channel) followed by a pointwise 1x1 conv (mixes channels).

**Why:** This is the limit of group convolution where g = C_in. The depthwise step captures spatial/temporal patterns independently per channel; the pointwise step mixes information across channels. The factorization reduces parameters from K * C_in * C_out to K * C_in + C_in * C_out. For sleep staging on edge devices, this is a key efficiency technique.

**Pattern:**
```
Input: (B, C, T)

Depthwise Conv1D(k=K, groups=C): (B, C, T)     # one filter per channel
Pointwise Conv1D(k=1, C -> C_out): (B, C_out, T) # channel mixing

Total params: K*C + C*C_out  (vs. K*C*C_out for standard)
```

**Seen in:** SleepSatelightFTC (DWConv2D in stem branches, expanding 16->32 channels; 0.47M params).

**Knobs:** Kernel size K of the depthwise conv. Whether to add BN + activation between the two steps or only after both. Expansion ratio (C_out / C).

---

## Self-Attention (Simple / Lightweight)

**Idea:** Let each position in a feature sequence compute attention weights over all other positions, using the features themselves as queries, keys, and values.

**Why:** Unlike convolutions which have a fixed receptive field, self-attention lets each position directly access any other position. For a feature map of T timesteps, this means a pattern at timestep 3 can directly influence the representation at timestep T-1. The quadratic cost (T^2) is acceptable when T is small (after aggressive downsampling from the stem). This is simpler than a full transformer -- no multi-head split, no FFN, just attention + residual.

**Pattern:**
```
Input: (B, T, D)

Q = K = V = Input         # or project: Q = X @ W_q, etc.
Scores = Q @ K^T / sqrt(D): (B, T, T)
Weights = Softmax(Scores):   (B, T, T)
Output = Weights @ V:        (B, T, D)

+ Residual: Output = Output + Input
+ BN or LayerNorm
```

**Seen in:** SleepSatelightFTC (self-attention blocks with residual + BN + dense expansion, 3 blocks per branch; ACC 85.7%).

**Knobs:** Whether to project Q, K, V or use raw features. Whether to add a dense/FFN block after attention (moving toward a full transformer). Normalization (BN vs. LayerNorm). Number of stacked blocks. Dropout on attention weights.

---

## Attention Pooling

**Idea:** Replace global average pooling with a learned weighted sum over the spatial dimension.

**Why:** GAP assumes all timesteps contribute equally to the final representation. This is almost never true in sleep EEG -- a K-complex at second 12 matters more than quiet background at second 25. Attention pooling learns a query vector that scores each timestep, then uses those scores as weights for a weighted sum. One extra parameter vector, substantially more expressive than GAP.

**Pattern:**
```
Input: (B, T, D)

Query: (D,) -- learned parameter
Scores = Input @ Query:     (B, T)
Weights = Softmax(Scores):  (B, T)
Output = (Weights.unsqueeze(-1) * Input).sum(dim=T): (B, D)
```

**Seen in:** SleePyCo (attention pooling collapses transformer output to single vector before classification; ACC 84.6%).

**Knobs:** Single query vs. multiple queries (multi-head pooling). Temperature scaling on scores. Whether to add a projection on the input before scoring.
