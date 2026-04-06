# Optimizer / Training

How the model learns: optimizer choice, learning rate scheduling, batch size, regularization, early stopping, multi-phase training pipelines, and data augmentation. These decisions shape convergence speed, generalization, and how well the model handles class imbalance and limited data.

---

## Adam with Fixed Learning Rate

**Idea:** Use the Adam optimizer at a constant learning rate for the entire training run.

**Why:** Adam adapts per-parameter learning rates using first and second moment estimates, so it handles the mixed scales of EEG features (slow delta waves vs. fast spindles) without manual tuning. A fixed rate works when the training run is short enough that the optimizer finds a good basin before overfitting, especially when paired with early stopping.

**Pattern:**
```
optimizer = Adam(lr=alpha, betas=(beta1, beta2), eps=epsilon)

# Typical ranges from sleep staging literature:
#   alpha:  1e-5  to  1e-3
#   beta1:  0.9
#   beta2:  0.999
#   eps:    1e-7  to  1e-8
```

**Seen in:** CSleepNet (lr=1e-5), CSCNN (lr=1e-3), MicroSleepNet (lr=1e-3), SleepSatelightFTC (lr=1e-3), TinySleepNet (lr=1e-4), SleePyCo (lr=1e-4), EnsembleNet CNN stage (lr=1e-3).

**Knobs:** The learning rate alpha is the dominant knob -- a 10x change matters more than anything else here. Lower rates (1e-5) pair with more epochs; higher rates (1e-3) pair with early stopping or LR decay. The betas are rarely changed from defaults.

---

## AdamW (Decoupled Weight Decay)

**Idea:** Use AdamW, which separates weight decay from the gradient-based update, rather than folding L2 into the loss.

**Why:** In standard Adam, L2 regularization interacts poorly with the adaptive learning rate -- heavily-updated parameters get less regularization than they should. AdamW fixes this by applying decay directly to the weights after the Adam step. This gives more consistent regularization across parameters, which matters when some filters learn fast (e.g., large-kernel slow-wave detectors) and others learn slowly.

**Pattern:**
```
optimizer = AdamW(lr=alpha, weight_decay=lambda, betas=(beta1, beta2))

# Train for N iterations (not epochs) with periodic validation
# Decouple the schedule: lr controls optimization speed, lambda controls model complexity
```

**Seen in:** ESSN (lr=5e-4, weight_decay=1e-4, 500K iterations).

**Knobs:** The weight_decay lambda is independent of lr here -- tune them separately. Iteration-based training (rather than epoch-based) pairs naturally with AdamW when sequence lengths vary across subjects.

---

## ReduceLROnPlateau

**Idea:** Monitor validation loss and reduce the learning rate by a fixed factor when improvement stalls.

**Why:** Sleep staging models often hit a plateau where the dominant classes (N2, Wake) are well-learned but minority classes (N1) still need refinement. Dropping the learning rate lets the optimizer make smaller, more precise updates to decision boundaries between confused stages without disrupting what it already knows.

**Pattern:**
```
scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=gamma,         # multiply lr by this on plateau
    patience=P,           # wait P checks before reducing
    min_lr=lr_floor       # stop reducing below this
)

# Typical: gamma=0.5, P=3, lr_floor=1e-7
```

**Seen in:** SCNet (factor=0.5, patience=3, min_lr=1e-7).

**Knobs:** Patience P controls how long you tolerate a plateau before reacting. Too short and you decay prematurely on noisy validation curves; too long and you waste compute. The factor gamma controls how aggressive each drop is -- 0.5 (halving) is standard but 0.1 (10x drop) works for sharper phase transitions.

---

## Step / Periodic LR Decay

**Idea:** Reduce the learning rate by a fixed factor at regular intervals regardless of validation performance.

**Why:** A deterministic schedule that doesn't depend on validation signal. Useful when validation is expensive (e.g., running HMM decoding or ensemble averaging) or when you want reproducible training dynamics. The optimizer explores broadly early on, then settles into fine-tuning as the rate drops.

**Pattern:**
```
# Every S epochs (or iterations), multiply lr by gamma
lr_t = lr_0 * gamma^(floor(t / S))

# Example: lr=0.01, gamma=0.5 every 100 epochs
#   epoch 0-99:   lr=0.01
#   epoch 100-199: lr=0.005
#   epoch 200-299: lr=0.0025
```

**Seen in:** EnsembleNet BiLSTM stage (lr=0.01, decay 0.5 every 100 epochs), ESSN (lr x 0.1 at validation checkpoints).

**Knobs:** The step interval S and decay factor gamma. Aggressive schedules (small S, small gamma) risk underfitting if the model hasn't converged before the rate gets too low. The total training budget (max epochs) should be set so you get at least 2-3 decay steps.

---

## Early Stopping

**Idea:** Stop training when validation performance hasn't improved for a set number of checks (patience), then restore the best model.

**Why:** Sleep staging datasets are modest in size (tens of subjects), so overfitting is a real threat -- especially for minority classes like N1. Early stopping acts as implicit regularization: it limits effective model capacity by controlling how long the model trains, without changing the architecture. It also makes training time adaptive to the dataset and fold.

**Pattern:**
```
best_metric = -inf
wait = 0
for each epoch (or validation check):
    if val_metric > best_metric:
        best_metric = val_metric
        save_checkpoint()
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        restore_checkpoint()
        break
```

**Seen in:** EnsembleNet (patience=30 CNN, 10 BiLSTM), ESSN (50 consecutive validation checks), MicroSleepNet (patience=10 epochs), SleepSatelightFTC (patience=5), SleePyCo (patience=20).

**Knobs:** Patience is the main knob. Short patience (5-10) works when validation signal is clean and you have many checks per epoch. Long patience (20-50) is needed when validation is noisy (small folds, high class imbalance) or when checks are infrequent. What you monitor matters too -- validation loss catches overfitting earlier than accuracy because accuracy can plateau while loss climbs.

---

## L2 Regularization / Weight Decay

**Idea:** Penalize large weights by adding a term proportional to the squared norm of the parameters to the loss (L2) or by directly shrinking weights each step (weight decay).

**Why:** EEG signals are noisy and sleep staging datasets have limited subjects. Without regularization, convolutional filters and dense layers can memorize subject-specific artifacts (electrode impedance patterns, individual alpha rhythms) instead of learning generalizable sleep features. L2 keeps weights small, which acts as a soft constraint toward simpler functions.

**Pattern:**
```
# L2 (in loss):
loss = task_loss + lambda * sum(p.pow(2) for p in model.parameters())

# Weight decay (in optimizer step):
p = p - lr * grad - lr * lambda * p

# Typical lambda range: 1e-6 to 1e-3
```

**Seen in:** CSleepNet (L2, lambda unspecified), EnsembleNet (L2, lambda=1e-4), MicroSleepNet (L2, lambda=1e-3), SCNet (L2, lambda=1e-3), TinySleepNet (L2, decay=1e-3), SleePyCo (decay=1e-6), ESSN (AdamW, decay=1e-4).

**Knobs:** The strength lambda. Small values (1e-6) are nearly invisible; large values (1e-3) aggressively constrain the model. The right value depends on model size -- small models (50K params) may not need much, while larger models (1M+) benefit from stronger regularization. When using AdamW, weight decay and L2-in-loss are different things -- don't stack both.

---

## Gradient Clipping

**Idea:** Cap the norm of gradients before applying the optimizer update to prevent exploding gradients.

**Why:** Recurrent models (LSTM, BiLSTM) processing long sequences can produce gradient spikes -- a single unusual epoch in a 15-epoch sequence can cause a huge gradient that destabilizes all learned weights. Clipping bounds the worst-case update size while leaving normal gradients untouched.

**Pattern:**
```
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=G)

# Typical G: 1.0 to 5.0
```

**Seen in:** TinySleepNet (max_norm=5.0).

**Knobs:** The threshold G. Too low and you slow convergence by truncating useful gradients; too high and you don't actually prevent blowups. Models without recurrence (pure CNNs) rarely need this. If you're using it and it's activating frequently, the learning rate is probably too high.

---

## Multi-Phase Training: Pretrain then Finetune

**Idea:** Train the model in distinct phases -- first learn representations (pretrain), then learn the task (finetune) -- potentially with different losses, data, and frozen layers.

**Why:** Sleep EEG has structure at multiple scales that is hard to learn in one shot. Contrastive pretraining can learn that "all N2 epochs look similar" without needing sequence context, then a second phase can learn temporal transitions. Splitting also avoids gradient conflicts -- the feature extractor and sequence model may fight over gradient magnitude if trained jointly from scratch.

**Pattern:**
```
# Phase 1: Pretrain feature extractor
backbone = train(backbone, loss=contrastive_or_reconstruction, data=single_epochs, lr=lr1, batch=B1)

# Phase 2: Finetune downstream
backbone.freeze()  # optional: full or partial freeze
model = Sequence(backbone, temporal_head)
model = train(model, loss=cross_entropy, data=epoch_sequences, lr=lr2, batch=B2)
```

**Seen in:** SleePyCo (CRL contrastive pretrain with batch=1024, then MTCL finetune with batch=64 and frozen backbone), EnsembleNet (90 CNN classifiers trained independently, then 10 BiLSTMs trained on frozen CNN outputs), SleepSatelightFTC (epoch model trained first, frozen, then transfer learning MLP for multi-epoch context).

**Knobs:** What to freeze and when -- full freeze is simplest but limits adaptation; partial freeze or gradual unfreezing can help. The pretrain objective (contrastive vs. reconstruction vs. classification) shapes what the backbone learns. Phase 1 often uses larger batches and different augmentation than Phase 2.

---

## Batch Size Selection

**Idea:** Choose the number of samples per gradient update, balancing gradient noise, memory, and implicit regularization.

**Why:** In sleep staging, "batch" can mean different things -- single epochs, multi-epoch sequences, or entire nights. Small batches (8-32) add gradient noise that can help escape poor minima and improve generalization on small datasets. Large batches (256-1024) give stable gradients needed for contrastive learning where you want many negatives. Memory is the practical constraint: a sequence of 200 epochs at 3000 samples each is large.

**Pattern:**
```
# Single-epoch models: batch = number of epochs
#   Range: 32 to 256
#
# Sequence models: batch = number of sequences (each containing N epochs)
#   Range: 8 to 64  (memory limited)
#
# Contrastive pretraining: batch = number of single epochs
#   Range: 512 to 1024  (need many negatives)
```

**Seen in:** ESSN (batch=8, sequences of 200 epochs), TinySleepNet (batch=20, sequences of 15), SleepSatelightFTC (batch=32), CSCNN (batch=64), SCNet (batch=64), EnsembleNet CNN (batch=128), MicroSleepNet (batch=200), CSleepNet (batch=256), SleePyCo CRL (batch=1024).

**Knobs:** Batch size and learning rate are coupled -- the linear scaling rule (double batch, double lr) is a rough guide. For sequence models, batch size is often dictated by GPU memory and sequence length. If using batch normalization, very small batches (< 16) can make BN statistics noisy -- consider group norm or layer norm instead.

---

## Data Augmentation for EEG

**Idea:** Apply random transformations to EEG epochs during training to increase effective dataset size and improve generalization.

**Why:** Sleep staging datasets typically have tens of subjects, but inter-subject variability (electrode placement, skull thickness, age, pathology) is high. Augmentation simulates some of this variability. The key constraint is that transformations must preserve sleep-stage-relevant features -- you can scale amplitude (simulates impedance variation) or add noise (simulates artifact), but you shouldn't time-warp at scales that distort sleep spindles or K-complexes.

**Pattern:**
```
# Applied sequentially with independent probability p per transform:
augmented = x
if random() < p: augmented = amplitude_scale(augmented, range=[1-a, 1+a])
if random() < p: augmented = time_shift(augmented, max_shift=s)
if random() < p: augmented = amplitude_shift(augmented, max_offset=o)
if random() < p: augmented = zero_mask(augmented, max_len=m)
if random() < p: augmented = add_gaussian_noise(augmented, std=sigma)
if random() < p: augmented = bandstop_filter(augmented, freq_range, bandwidth)

# Typical p=0.5 per transform
```

**Seen in:** SleePyCo CRL (6 transforms: amplitude scaling, time shift, amplitude shift, zero-masking, Gaussian noise, band-stop filter, each p=0.5), TinySleepNet (signal shift +/-10%, sequence offset 0-5 epochs).

**Knobs:** Which transforms to include and how aggressive each one is. Amplitude scaling and Gaussian noise are safe -- they simulate real recording variation. Zero-masking teaches robustness to dropout artifacts. Band-stop filtering simulates notch filters or frequency-specific noise. Time shift magnitude should stay well below the epoch boundary (a 10% shift on 30s = 3s, which is reasonable). Augmentation is typically used only during pretraining or the feature learning phase, not during sequence-level training.

---

## Mixed Precision Training (FP16)

**Idea:** Use half-precision (16-bit) floating point for forward/backward passes while keeping a master copy of weights in FP32.

**Why:** Halves memory usage for activations, which directly translates to fitting longer sequences or larger batches on the same GPU. Sleep sequence models processing 200 epochs at once benefit enormously. The speed gain (1.5-2x on modern GPUs) is a bonus. Loss scaling prevents underflow in FP16 gradients.

**Pattern:**
```
scaler = GradScaler()
with autocast(dtype=float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Seen in:** ESSN (FP16, processing 200-epoch sequences).

**Knobs:** Whether to use it at all -- small models on short sequences don't need it. Some operations (e.g., softmax in attention, loss computation) should stay in FP32 for numerical stability. Modern frameworks handle this automatically with autocast, but custom loss functions may need explicit dtype management.

---

## Iteration-Based Training (vs. Epoch-Based)

**Idea:** Define training duration and validation frequency in terms of gradient steps (iterations) rather than passes over the dataset (epochs).

**Why:** In sleep staging, "one epoch of data" varies wildly across folds and datasets -- a LOSO fold with 19 subjects has much more data than one with 4. Iteration-based training keeps the optimization trajectory consistent regardless of fold size. It also pairs naturally with early stopping based on validation checks at fixed intervals.

**Pattern:**
```
for iter in range(max_iterations):
    batch = sample_from_training_set()
    step(batch)

    if iter % val_interval == 0:
        val_metric = validate()
        check_early_stopping(val_metric)

# Example: 500K iterations, validate every 100 steps
# Example: validate every 50 iterations (SleePyCo CRL), every 500 iterations (MTCL)
```

**Seen in:** ESSN (500K iterations, validate every 100), SleePyCo (CRL: validate every 50 iterations, MTCL: every 500 iterations).

**Knobs:** Max iterations and validation interval. The validation interval controls the granularity of your early stopping -- too frequent wastes compute on validation, too infrequent means you overshoot the best checkpoint. Scale both with dataset size: larger datasets need more iterations and can afford less frequent validation.
