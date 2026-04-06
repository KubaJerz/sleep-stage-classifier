# Loss

What the model optimizes for -- the loss decides how prediction errors translate into gradient signal. In sleep staging, the central tensions are class imbalance (N2 dominates, N1 is rare and confusable), hard-sample misclassification (transition epochs between stages), and whether auxiliary objectives can inject useful inductive bias beyond the main classification target.

---

## Cross-Entropy

**Idea:** Standard categorical cross-entropy over the C sleep stage classes -- the default baseline loss.

**Why:** CE directly maximizes the log-likelihood of the correct class. It works well when classes are roughly balanced or when imbalance is handled elsewhere (e.g., wake trimming in preprocessing). Most papers start here and only add complexity if CE alone leaves specific stages underperforming.

**Pattern:**
```
logits: (B, C)          -- raw model output
targets: (B,)           -- integer class labels

L = -mean over B [ log softmax(logits)[target] ]
```

**Seen in:** CSCNN+HMM (ACC 84.6%), EnsembleNet (ACC 87.07%), ESSN (as primary term, ACC 87.1%), SCNet (ACC 86.1%), SleepSatelightFTC (ACC 85.7%), SleePyCo MTCL step (ACC 84.6%), TinySleepNet (as base before weighting, ACC 85.4%).

**Knobs:** Label smoothing epsilon (softens one-hot targets, prevents overconfident predictions). Whether to apply per-epoch or per-sequence (sequence models predicting multiple epochs at once sum CE across the sequence length).

---

## Class-Weighted Cross-Entropy

**Idea:** Multiply each sample's CE contribution by a weight that depends on its class, giving underrepresented stages more influence on the gradient.

**Why:** Sleep datasets are heavily skewed -- N2 often exceeds 35% of epochs while N1 sits below 10%. Without weighting, the model can achieve decent overall accuracy by mostly ignoring N1. Class weights counteract this by making an N1 misclassification cost more than an N2 misclassification. The weights can come from inverse class frequency, manual tuning, or a log-dampened frequency formula.

**Pattern:**
```
logits: (B, C)
targets: (B,)
weights: (C,)           -- one weight per class, e.g., inversely proportional to frequency

L = -mean over B [ w[target] * log softmax(logits)[target] ]
```

**Seen in:** TinySleepNet (N1=1.5, others=1.0; MF1 80.5% on EDF-20), Micro SleepNet (weights [1,2,1,1,1] on SHHS for N1 boost; ACC 83.3%).

**Knobs:** How to compute weights -- inverse frequency, square-root inverse frequency, log-inverse frequency, or manual per-class values. The degree of boost matters: too aggressive and the model overpredict the minority class. Whether to normalize weights so they sum to C (keeps gradient magnitude stable).

---

## Focal Loss

**Idea:** Down-weight easy (well-classified) samples and focus the loss on hard samples where the model is uncertain or wrong.

**Why:** Even within the majority class, many epochs are trivially classified (deep N2, stable W). These easy samples dominate the gradient and dilute the learning signal from genuinely ambiguous epochs -- transition points between N1/N2, or N1/W boundaries. Focal loss adds a modulating factor (1 - p_t)^gamma that suppresses the loss when the model is already confident, letting hard cases drive learning.

**Pattern:**
```
logits: (B, C)
targets: (B,)
gamma: scalar            -- focusing parameter, typically 1.0-3.0

p_t = softmax(logits)[target]
L = -mean over B [ (1 - p_t)^gamma * log(p_t) ]
```

**Seen in:** SCANSleepNet (combined with class weighting as WFCE; ACC 85.52%, MF1 78.31%).

**Knobs:** Gamma -- at gamma=0 this is plain CE; gamma=2 is a common starting point. Combining with class weights (multiply focal term by class weight) gives weighted focal CE. Alpha balancing factor (per-class scaling applied alongside gamma).

---

## Weighted Focal Cross-Entropy

**Idea:** Combine focal loss with log-frequency class weighting into a single loss that simultaneously handles class imbalance and hard-sample focus.

**Why:** Class weighting alone treats all samples from a rare class equally, even trivially-correct ones. Focal loss alone ignores the prior frequency of classes. Combining both means: rare classes get a baseline boost (weighting), and within every class, hard samples get additional focus (focal modulation). The log-frequency weighting is smoother than raw inverse frequency and avoids extreme weight ratios.

**Pattern:**
```
logits: (B, C)
targets: (B,)
gamma: scalar
class_counts: (C,)

w_c = log(total / class_counts[c])     -- log-frequency class weight
p_t = softmax(logits)[target]
L = -mean over B [ w[target] * (1 - p_t)^gamma * log(p_t) ]
```

**Seen in:** SCANSleepNet (ACC 85.52%, MF1 78.31%, kappa 0.80 on Fpz-Cz).

**Knobs:** Gamma (focal strength). Weight formula (log vs sqrt-inverse vs manual). Whether to normalize weights. Whether log is base-e or base-10 (changes the weight spread).

---

## Confusion-Subset Auxiliary Loss (N1 Structure Loss)

**Idea:** Add a secondary CE loss computed over only the subset of classes that are most confused with each other, forcing the model to sharpen its decisions within that confusion cluster.

**Why:** N1 is systematically confused with W and N2 -- these three stages share similar EEG amplitude and frequency content. The main CE loss spreads its gradient across all 5 classes, but the N1/W/N2 discrimination problem needs concentrated attention. By extracting the logits for {W, N1, N2}, re-normalizing with softmax over just those three classes, and computing a separate CE, the model receives an additional gradient that directly targets the hardest confusion boundary.

**Pattern:**
```
logits: (B, C)                        -- full 5-class output
targets: (B,)
subset: list of class indices         -- e.g., [W, N1, N2]
lambda_aux: scalar                    -- auxiliary loss weight

L_main = CE(logits, targets)
logits_sub = logits[:, subset]        -- (B, |subset|)
targets_sub = remap targets to subset indices (ignore samples outside subset)
L_aux = CE(softmax(logits_sub), targets_sub)   -- only on samples belonging to subset classes

L = L_main + lambda_aux * L_aux
```

**Seen in:** ESSN (lambda=1.5e-2, subset={W, N1, N2}; ACC 87.1%, MF1 81.4%).

**Knobs:** Which subset of classes to target -- {W, N1, N2} is the obvious one for sleep staging, but a confusion matrix analysis might reveal others. Lambda weight for the auxiliary term. Whether to apply the auxiliary loss to all samples or only to samples whose true label is in the subset.

---

## Supervised Contrastive Loss (Pretraining)

**Idea:** Before training the classifier, pretrain the feature backbone so that embeddings of same-stage epochs cluster together and different-stage epochs are pushed apart in representation space.

**Why:** CE only cares about the decision boundary -- it does not explicitly shape the geometry of the learned feature space. Contrastive pretraining produces embeddings where all W epochs form a tight cluster, all N2 epochs form another, etc. This structure makes the downstream classifier's job easier, especially for rare classes that get little gradient signal from CE alone. It also improves generalization because the backbone learns features based on inter-class similarity structure, not just boundary placement.

**Pattern:**
```
-- Pretraining step (backbone + projection head) --
epoch_i: (1, T)                        -- single epoch
z_i = project(backbone(epoch_i))       -- (d_z,)  e.g., d_z=128
tau: scalar                            -- temperature, controls separation sharpness

For anchor z_i with label y_i in a batch of N samples:
  positives = {z_j : y_j == y_i, j != i}
  L_i = -mean over positives [ log( exp(z_i . z_j / tau) / sum_k!=i exp(z_i . z_k / tau) ) ]
L = mean over all anchors [ L_i ]

-- Fine-tuning step --
Freeze backbone, discard projection head, train classifier head with CE.
```

**Seen in:** SleePyCo CRL step (tau=0.07, d_z=128, batch=1024; downstream ACC 84.6%, MF1 79.0%).

**Knobs:** Temperature tau (lower = harder separation, typical range 0.05-0.1). Projection head dimension d_z. Batch size (contrastive loss needs large batches for diverse negatives). Whether to use supervised contrastive (uses labels) or self-supervised contrastive (uses augmentations only). How many epochs to pretrain before switching to classification fine-tuning.

---

## Multi-Level Loss (Feature Pyramid)

**Idea:** Compute the classification loss at multiple spatial resolutions from a feature pyramid and sum them, so every pyramid level receives direct supervision.

**Why:** Feature pyramids capture information at different temporal granularities -- fine-grained features at early levels, abstract features at deeper levels. If only the final output is supervised, intermediate levels get diluted gradients. Applying CE at each level forces every scale to produce independently useful representations and prevents later layers from compensating for uninformative earlier ones.

**Pattern:**
```
pyramid levels: F_1, F_2, ..., F_K     -- feature maps at K resolutions
shared_classifier: maps each F_k -> (B, C)

o_k = classifier(F_k)                  -- (B, C) logits at level k
L = sum over k [ CE(o_k, targets) ]

-- At inference, sum logits: y_hat = argmax( sum_k o_k )
```

**Seen in:** SleePyCo MTCL step (K=3 pyramid levels with shared transformer classifier; ACC 84.6%).

**Knobs:** Number of pyramid levels K. Whether to weight each level's loss equally or decay based on resolution. Whether the classifier is shared across levels or independent. Summing logits vs. averaging probabilities at inference.

---

## Logcosh Loss

**Idea:** Use log(cosh(error)) as the loss function instead of squared error or cross-entropy -- a smooth approximation to L1 that behaves like L2 for small errors and like L1 for large errors.

**Why:** Sleep stage labels have noise -- inter-rater agreement among human scorers is only about 80-85%. Transition epochs between stages are inherently ambiguous. Logcosh is less sensitive to outlier labels than squared error (it does not blow up for large errors) while still being smooth and differentiable everywhere (unlike raw L1). This makes training more stable when label noise is present.

**Pattern:**
```
predictions: (B, C)      -- typically after softmax, treated as regression targets
targets: (B, C)          -- one-hot encoded

error = predictions - targets
L = mean over B [ sum over C [ log(cosh(error)) ] ]

-- Approximation: behaves like 0.5 * error^2 for small |error|,
--                behaves like |error| - log(2) for large |error|
```

**Seen in:** CSleepNet (best among 4 losses on children's dataset: ACC 83.06%, F1 76.50%).

**Knobs:** Whether to apply on raw logits or post-softmax outputs. Combining with class weighting. Compared against CE, hinge, and Poisson in CSleepNet -- CE was close but logcosh won on the pediatric dataset where label noise may be higher.

---

## L2 Regularization (Weight Decay)

**Idea:** Add a penalty proportional to the squared magnitude of all model weights, discouraging large weight values.

**Why:** Sleep staging models are trained on relatively small datasets (a few hundred nights, tens of thousands of epochs). Without regularization, models with many parameters memorize subject-specific artifacts. L2 decay pushes weights toward zero, acting as a soft constraint on model complexity. It works alongside the primary loss -- it does not change what the model optimizes for, but constrains how aggressively it fits.

**Pattern:**
```
theta: all model parameters
lambda: scalar                -- regularization strength

L_total = L_primary + lambda * sum(theta_i^2)

-- Equivalently, implemented as weight_decay parameter in optimizer (e.g., AdamW)
```

**Seen in:** CSleepNet (lambda unspecified), EnsembleNet (lambda=1e-4), Micro SleepNet (lambda=1e-3), SCNet (lambda=1e-3, applied to conv layers), TinySleepNet (lambda=1e-3), SleePyCo (decay=1e-6).

**Knobs:** Lambda value (typical range 1e-6 to 1e-2). Whether to apply to all parameters or exclude biases and batch norm. Whether to use true L2 (add to loss) or decoupled weight decay (AdamW-style, which behaves differently with adaptive optimizers). Per-layer lambda values if different parts of the network need different regularization strength.
