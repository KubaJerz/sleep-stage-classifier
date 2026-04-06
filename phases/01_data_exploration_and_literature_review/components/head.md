# Head

The head collapses a learned feature representation into class probabilities. Everything upstream extracts and refines features; the head makes the final decision. The design choice here is mostly about *how* to aggregate spatial/temporal dimensions that remain in the feature map, and how much nonlinearity to put between features and logits.

---

## GAP + Linear

**Idea:** Global average pooling squashes the entire temporal dimension, then a single linear layer projects to class logits.

**Why:** By the time features reach the head, good upstream processing has already done the heavy lifting. GAP removes the temporal axis without adding parameters (no flatten explosion), and a single linear layer keeps the head from overfitting. This is the "let the backbone do the work" philosophy -- the head is intentionally weak so all discriminative pressure falls on the feature extractor. Also acts as a mild regularizer since it forces the network to spread class-relevant information across all time positions rather than concentrating it in one spot.

**Pattern:**
```
Input:   (B, C, T)
GAP:     (B, C, T) -> mean over T -> (B, C)
Dropout: (B, C) -> (B, C)           [optional, p ~ 0.5]
Linear:  (B, C) -> (B, num_classes)
Softmax: (B, num_classes)
```

**Seen in:** MicroSleepNet (ACC 82.8%, 48K params), SingleChannelNet (ACC 86.1%)

**Knobs:** Whether to include dropout before the linear layer (SCNet uses 0.5, MicroSleepNet does not). Whether the pooling is pure average or includes max-pool information (SCNet's M-APooling concatenates max and avg, doubling the channel dimension before the linear layer).

---

## Flatten + Linear

**Idea:** Flatten the entire feature map into a single vector and project directly to class logits with one dense layer.

**Why:** The most minimal possible head -- no hidden layers, no pooling tricks. Works when the feature map is already small enough that flattening does not create an enormous parameter count. The implicit assumption is that spatial position in the final feature map still carries meaning (unlike GAP, which discards it), so the linear layer can learn position-specific weights.

**Pattern:**
```
Input:   (B, C, T)
Flatten: (B, C * T)
Linear:  (B, C * T) -> (B, num_classes)
Softmax: (B, num_classes)
```

**Seen in:** CSCNN + HMM (ACC 84.6% on Fpz-Cz, flatten 71x128=9088 -> 5)

**Knobs:** Whether the upstream feature map is small enough for this to be practical. If C*T is large (thousands+), the parameter count in the linear layer dominates the model and overfitting risk grows -- at that point, GAP or an MLP bottleneck is usually better.

---

## MLP Head (Multi-Layer FC)

**Idea:** A small fully-connected network with one or more hidden layers between the feature representation and the class logits.

**Why:** Adds learnable nonlinear capacity to the head itself. Useful when the upstream features are high-dimensional or when different feature dimensions interact in ways a single linear layer cannot capture. The narrowing structure (wide -> narrow -> classes) acts as an information bottleneck, forcing the network to compress to the most class-relevant dimensions. Dropout between layers regularizes the head independently of the backbone.

**Pattern:**
```
Input:    (B, D)                         [D from flatten or pooling]
Linear:   (B, D) -> (B, H1)             [H1 << D]
Act+Drop: ReLU/SiLU + Dropout(p)
Linear:   (B, H1) -> (B, H2)            [optional second hidden layer, H2 < H1]
Act+Drop: ReLU/SiLU + Dropout(p)
Linear:   (B, H2) -> (B, num_classes)
Softmax:  (B, num_classes)
```

**Seen in:** CSleepNet (1024 -> 256 -> 64 -> 5, ACC 86.4%), SCANSleepNet (Flatten -> FC -> FC -> 5, ACC 85.5%), Raspberry Pi CNN (1500 -> 10 -> 2, deployed on RPi3), SleepSatelightFTC (896 -> 300 -> 5, ACC 85.7%)

**Knobs:** Number of hidden layers (1-3 typical). Hidden dimension schedule (gradual narrowing vs. aggressive bottleneck). Activation function (ReLU is standard; SiLU/Mish in more recent work). Dropout rate per layer (0.3-0.5 common). Whether the input comes from flatten or pooling.

---

## Per-Timestep Projection

**Idea:** Apply the same linear layer independently at every time step in a sequence, producing one class prediction per epoch.

**Why:** This is the natural head for sequence-to-sequence architectures where the backbone (LSTM, Transformer, etc.) outputs a hidden state per epoch and you want a prediction for each one. The classifier is shared across all positions, so it learns a single decision boundary applied uniformly. No temporal collapsing happens in the head -- that responsibility lives entirely in the backbone. This is also the most parameter-efficient head since it is just one small linear layer, applied N times.

**Pattern:**
```
Input:   (B, N, H)                      [N epochs, H hidden dim per epoch]
Dropout: (B, N, H)                      [optional]
Linear:  (B, N, H) -> (B, N, num_classes)   [shared weights across N]
Softmax: (B, N, num_classes)
```

**Seen in:** TinySleepNet (LSTM 128 -> 5 at each of 15 timesteps, ACC 85.4%), ESSN (128 -> 5 at each of 200 timesteps, ACC 87.1%)

**Knobs:** Whether to add dropout before the projection. Whether the linear layer is truly a single layer or a small MLP applied per timestep (ESSN uses Linear + SiLU + Dropout before the final projection). Sequence length N (TinySleepNet uses 15, ESSN uses 200 -- this is really an architectural choice upstream but it determines how many predictions the head produces).

---

## Attention Pooling + Linear

**Idea:** Instead of averaging all time positions equally (GAP), learn a weighted combination of positions via an attention mechanism, then project the weighted sum to class logits.

**Why:** Not all time steps contribute equally to the classification. Attention pooling lets the model focus on the most informative positions -- a spindle burst matters more than quiet background. This is a middle ground between GAP (treats all positions equally) and flatten (treats each position independently). It collapses the temporal axis like GAP but with learned, input-dependent weights.

**Pattern:**
```
Input:    (B, T, H)
Query:    learnable vector q of dim H     [or projected from input]
Scores:   (B, T) = softmax(Input @ q / sqrt(H))
Weighted: (B, H) = sum(Scores * Input, dim=T)
Linear:   (B, H) -> (B, num_classes)
```

**Seen in:** SleePyCo (attention pooling after Transformer encoder at each pyramid level, ACC 84.6%)

**Knobs:** How the query is formed (fixed learnable vector vs. derived from input). Temperature scaling on the attention scores. Whether to use multi-head attention pooling or single-head.

---

## Multi-Scale Logit Fusion

**Idea:** Produce independent logit vectors from multiple feature scales (e.g., a feature pyramid), then sum or average them before the final argmax.

**Why:** Different sleep features live at different time scales -- slow waves span seconds, spindles are sub-second, K-complexes are somewhere in between. A feature pyramid captures these at different resolutions. Rather than forcing a single head to reconcile all scales, each scale gets its own classifier and the final decision is a vote across scales. This is implicit ensembling built into the architecture.

**Pattern:**
```
Features at scale i: (B, T_i, H)
Per scale:
  Pool/Attend:  (B, T_i, H) -> (B, H)
  Linear:       (B, H) -> (B, num_classes)   [separate or shared weights]

Fusion: logits = sum(logits_1, logits_2, ..., logits_K)
Prediction: argmax(logits)
```

**Seen in:** SleePyCo (3 pyramid levels with shared Transformer + attention pool + FC, logits summed, ACC 84.6%)

**Knobs:** Number of scales (SleePyCo uses 3). Whether the per-scale classifiers share weights or are independent (SleePyCo shares). Fusion operation (sum, mean, learned weighted sum). Whether fusion happens at the logit level or the probability level (logit-level sum is equivalent to a geometric mean of probabilities).

---

## Ensemble Averaging

**Idea:** Train multiple independent models (or model branches) and average their output probabilities before taking argmax.

**Why:** Reduces variance. Individual models may overfit to different patterns or make uncorrelated errors. Averaging smooths these out. Especially useful when the base models are small or the training procedure is stochastic (different initializations, different data splits). The cost is linear in the number of ensemble members at both training and inference time.

**Pattern:**
```
K independently trained models, each producing:
  Model_k: input -> (B, num_classes)   [softmax probabilities]

Ensemble: probs = (1/K) * sum(probs_1, ..., probs_K)
Prediction: argmax(probs)
```

**Seen in:** EnsembleNet (10 BiLSTMs averaged, ACC 87.1% on EDF-20)

**Knobs:** Number of ensemble members K (EnsembleNet uses 10). Whether members differ in architecture, initialization, or training data. Whether to average probabilities (soft voting) or class predictions (hard voting / majority vote). Whether the ensemble is over full models or just the temporal stage (EnsembleNet freezes the CNN feature extractor and only ensembles the BiLSTM sequence models).

---

## Bottleneck Fusion Head

**Idea:** Concatenate feature vectors from multiple branches (e.g., time-domain + frequency-domain), compress through a narrow FC layer, then project to classes.

**Why:** When the model has parallel processing streams that each produce their own feature vector, the head needs to combine them. Naive concatenation + linear would work but the combined dimension can be large. A bottleneck layer forces the network to learn a compact joint representation from both streams before classification. The bottleneck dimension also becomes a reusable embedding (SleepSatelightFTC reuses the 300-dim bottleneck for transfer learning across epochs).

**Pattern:**
```
Branch A output: (B, D_a)
Branch B output: (B, D_b)
Concat:          (B, D_a + D_b)
Linear:          (B, D_a + D_b) -> (B, D_bottleneck)   [D_bottleneck << D_a + D_b]
Activation:      ReLU / SiLU
Linear:          (B, D_bottleneck) -> (B, num_classes)
Softmax:         (B, num_classes)
```

**Seen in:** SleepSatelightFTC (time 640 + freq 256 = 896 -> 300 -> 5, ACC 85.7%; bottleneck reused for multi-epoch transfer)

**Knobs:** Bottleneck dimension (controls compression ratio and reusability). Number of branches being fused. Whether to add dropout or batch norm in the bottleneck. Whether the bottleneck embedding is reused downstream (e.g., for transfer learning or multi-epoch context).
