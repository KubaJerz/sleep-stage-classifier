# General Model Structure

## Input

Multiple consecutive epochs fed as a sequence. The model classifies **only the last (current) epoch**. Context is **causal** — prior epochs and the current epoch only, never future epochs. The number of context epochs is a design choice.

## Architecture

### 1. Stem (feature extraction)

Extracts features from input epochs. Epochs can be processed individually or jointly. Weight sharing across epochs is a design choice.

### 2. Backbone (temporal context)

Aggregates information across the epoch sequence. 

### 3. Head (classification)

Maps the current-epoch representation to a probability distribution over 5 classes.
