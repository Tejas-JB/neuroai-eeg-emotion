# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# neuroai-eeg-emotion
## Emotion Classification from EEG Brain Data Using Spiking Neural Networks

---

## 1. EXECUTIVE SUMMARY

**Project name:** neuroai-eeg-emotion

**One-liner:** Emotion classification from real EEG brain data using Spiking Neural Networks — because brain data IS spikes, so why would you use a non-spiking model to process it?

**Thesis:** Spiking Neural Networks (SNNs) are biologically faithful models of neural computation. EEG data captures real neural activity. Using an SNN to decode EEG emotion signals is more natural, more energy-efficient, and produces interpretable spike-based representations that traditional deep learning cannot.

**Author:** Tejas J Bharadwaj — CMU SCS, Class of 2030

**License:** MIT (maximum openness)

**Repository:** github.com/[tejas-handle]/neuroai-eeg-emotion

---

## 2. OBJECTIVES & SUCCESS CRITERIA

### 2.1 Primary Objectives

1. Build a complete, reproducible pipeline that classifies emotions (positive, neutral, negative) from EEG data using an SNN built with snnTorch
2. Compare SNN performance against a baseline MLP with identical parameter budget
3. Produce publication-quality visualizations of internal SNN spike dynamics per emotion class
4. Package everything as a clean, well-documented open source repository

### 2.2 Success Criteria

| Metric | Target |
|--------|--------|
| Pipeline completeness | End-to-end: raw .mat → spike encoding → SNN → emotion prediction → visualization |
| Synthetic data pipeline | Full pipeline runs on synthetic data with zero external data dependencies |
| Real data pipeline | Full pipeline runs on SEED dataset DE features |
| Baseline comparison | SNN accuracy reported alongside MLP accuracy on identical train/test splits |
| Visualization count | Minimum 4 distinct visualization types (spike raster, confusion matrix, accuracy comparison, membrane traces) |
| Code quality | All Python files pass flake8 linting, consistent style |
| Documentation | README with quick-start, architecture explanation, results, and author attribution |
| Reproducibility | Any user can clone, install, and run on synthetic data in under 5 minutes |

### 2.3 Non-Goals

- Real-time EEG processing or streaming inference
- Deployment to neuromorphic hardware (Loihi, SpiNNaker)
- Raw EEG signal processing (we use pre-extracted DE features)
- Mobile or web deployment
- Achieving state-of-the-art accuracy (the goal is a clean demonstration, not a benchmark chase)

---

## 3. BACKGROUND & MOTIVATION

### 3.1 Why SNNs for EEG?

Traditional deep learning models (CNNs, LSTMs, Transformers) process EEG data as continuous-valued tensors. But biological neurons communicate through discrete spikes — brief electrical impulses. EEG data captures the aggregate electrical activity of millions of neurons firing spikes. Using an SNN to process EEG data respects this biological reality.

Key advantages:
- **Biological plausibility:** SNN neurons (Leaky Integrate-and-Fire) accumulate input over time, fire when a threshold is reached, and reset — mirroring real neurons
- **Energy efficiency:** The combra-lab TMLR paper demonstrated 95% less energy consumption vs DNNs at comparable accuracy on EEG decoding tasks
- **Sparsity:** SNNs are inherently sparse — most neurons are silent at any given timestep, meaning computation only happens where it matters
- **Interpretability:** Spike raster plots directly show WHEN and WHERE the network activates for different emotion classes, providing visual insight that activation maps in traditional NNs cannot match

### 3.2 Gap in Existing Work

| What exists | What's missing |
|-------------|---------------|
| combra-lab/snn-eeg: SNN + EEG decoding (motor imagery only) | Nobody has applied SNN to EEG emotion classification |
| Dozens of CNN/LSTM repos for EEG emotion classification (DEAP, SEED) | All use traditional non-spiking architectures |
| snnTorch tutorials covering SNN fundamentals | No tutorial or example for EEG applications |
| SEED dataset with pre-extracted DE features | No clean public repo combining SEED + SNN |

This project fills the gap: SNN + EEG + emotion classification in a clean, reproducible, open-source package.

### 3.3 References

- Zheng & Lu, "Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks", IEEE Trans. Autonomous Mental Development, 2015 (SEED dataset paper)
- combra-lab, "Decoding EEG With Spiking Neural Networks on Neuromorphic Hardware", TMLR (SNN EEG decoding)
- snnTorch documentation and tutorials (snntorch.readthedocs.io)

---

## 4. DATASET: SEED

### 4.1 Overview

| Property | Value |
|----------|-------|
| Name | SEED (SJTU Emotion EEG Dataset) |
| Source | Shanghai Jiao Tong University BCMI Lab |
| Subjects | 15 |
| Sessions per subject | 3 (45 sessions total) |
| EEG channels | 62 |
| Sampling rate | 1000 Hz (downsampled to 200 Hz) |
| Stimuli | 15 film clips per session designed to elicit emotions |
| Emotion classes | 3: Positive (+1), Neutral (0), Negative (-1) |
| License | Academic/research use only, non-commercial, do not redistribute |

### 4.2 Pre-Extracted Features (What We Use)

The SEED dataset provides pre-extracted **Differential Entropy (DE)** features. These are the recommended starting point and what our pipeline consumes.

**DE feature specification:**
- 62 EEG channels
- N time windows per trial (variable, depends on clip length)
- 5 frequency bands per channel: delta (1-4 Hz), theta (4-8 Hz), alpha (8-14 Hz), beta (14-31 Hz), gamma (31-50 Hz)
- Feature vector per time window: 62 channels × 5 bands = **310 features**
- Format: MATLAB .mat files, loadable with scipy.io.loadmat or h5py

### 4.3 Labels

- Labels are defined by the SEED experimental protocol, NOT stored in a separate label.mat file
- The label order per session is the SAME for all subjects and all sessions: `[1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]`
  - Mapped to our classes: `[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]` (where 0=negative, 1=neutral, 2=positive)
- This maps to 15 trials per session (15 film clips)
- The label-to-trial mapping is confirmed in SEED_stimulation.xlsx and the original SEED paper (Zheng & Lu, 2015)
- Labels may also be embedded inside each .mat file under a key like `label` — check for this and use it if present, otherwise use the hardcoded protocol labels above

### 4.4 Expected File Structure

```
data/
├── ExtractedFeatures_1s/          # ← PRIMARY INPUT (download this folder)
│   ├── 1_20131027.mat             # Subject 1, Session 1 (Oct 27, 2013)
│   ├── 1_20131030.mat             # Subject 1, Session 2
│   ├── 1_20131107.mat             # Subject 1, Session 3
│   ├── 2_20140404.mat             # Subject 2, Session 1
│   ├── 2_20140413.mat             # Subject 2, Session 2
│   ├── 2_20140419.mat             # Subject 2, Session 3
│   ├── ...                        # Pattern: {subject_id}_{YYYYMMDD}.mat
│   └── 15_XXXXXXXX.mat            # Subject 15, Session 3
├── channel-order.xlsx             # 62-channel electrode positions (10 KB)
├── SEED_stimulation.xlsx          # Film clip details + emotion labels (26 KB)
└── subject-id-gender-seed.txt     # Subject demographics (66 bytes)
```

**File naming convention:** `{subject_id}_{session_date}.mat`
- Subject IDs: 1 through 15
- Each subject has exactly 3 sessions (3 dates)
- 45 total .mat files (15 subjects × 3 sessions) + possibly 2 extra sessions = 47 files total
- Each file is approximately 60-61 MB
- Total folder size: ~2.8 GB

**Internal .mat file structure (CRITICAL — must be inspected at runtime):**
- Each .mat file contains 15 trial keys (one per film clip)
- Key naming pattern is likely: `de_LDS1`, `de_LDS2`, ..., `de_LDS15` OR `de_movingAv1`, ..., `de_movingAv15` OR similar prefixed keys
- Each trial value is a numpy array of shape approximately (N_windows, 62, 5) where:
  - N_windows = number of 1-second windows in that film clip (varies per clip, roughly 50-240)
  - 62 = EEG channels
  - 5 = frequency bands (delta, theta, alpha, beta, gamma)
- The data loader MUST: load one file, print ALL keys, print the shape of the first non-metadata key, and adapt accordingly
- Files may also contain metadata keys like `__header__`, `__version__`, `__globals__` — skip these
- Try `scipy.io.loadmat()` first; if it fails, fall back to `h5py.File()` (MATLAB v7.3 / HDF5 format)

**Note:** The exact key names WILL vary. The data loader must discover keys dynamically, not hardcode them.

### 4.5 Data Processing Pipeline

1. Load .mat file with scipy.io.loadmat (fall back to h5py if v7.3)
2. Extract DE feature matrices per trial
3. Load corresponding emotion labels
4. Reshape each trial's features into time windows of shape (N_windows, 310)
5. Assign the trial's emotion label to all windows in that trial
6. Aggregate across all trials/sessions
7. Output: X (total_samples, 310), y (total_samples,) with values in {0, 1, 2}
8. Stratified train/test split (80/20)

### 4.6 Synthetic Data Fallback

A synthetic data generator must produce data with the identical interface:
- X: (N_samples, 310) random features in [0, 1]
- y: (N_samples,) balanced labels in {0, 1, 2}
- Default: 3000 samples (1000 per class)
- Purpose: enables full pipeline development and testing without SEED data
- Expected accuracy on synthetic data: ~33% (random chance for 3 classes) — this is correct behavior, not a bug

---

## 5. TECHNICAL ARCHITECTURE

### 5.1 System Overview

```
Input (.mat files)
    │
    ▼
[data_loader.py] ── Load DE features, extract windows, assign labels
    │
    ▼
X (samples, 310), y (samples,)
    │
    ▼
[spike_encoder.py] ── Normalize → Rate coding → Spike trains
    │
    ▼
Spike trains (T, batch, 310)    Raw features (batch, 310)
    │                                │
    ▼                                ▼
[snn_model.py]                 [baseline_mlp.py]
    │                                │
    ▼                                ▼
Spike counts (batch, 3)        Logits (batch, 3)
    │                                │
    ▼                                ▼
[train.py] ── Training loop, loss, optimizer, logging
    │
    ▼
[evaluate.py] ── Metrics, confusion matrix, classification report
    │
    ▼
[visualize.py] ── Spike rasters, membrane traces, comparison charts
    │
    ▼
results/ ── PNG outputs
```

### 5.2 Spike Encoding (spike_encoder.py)

**Purpose:** Convert continuous DE features into binary spike trains suitable for SNN processing.

**Primary method: Rate coding**
- Normalize all features to [0, 1] range (min-max per feature across training set)
- Use snnTorch's `spikegen.rate()` function
- Higher feature values → higher probability of spiking at each timestep
- Output: binary tensor of shape (T, batch_size, 310) where T = number of timesteps

**Secondary method: Delta coding**
- Encode the change in feature values over time
- Spikes are generated when the feature change exceeds a threshold
- Useful for temporal EEG patterns
- Implemented as an alternative via a flag

**Parameters:**
- T (timesteps): 25 (default, configurable)
- Normalization: min-max to [0, 1]
- Encoding method: "rate" (default) or "delta"

**Quality checks:**
- Sparsity should be between 40-95% (not all zeros, not all ones)
- Mean firing rate should vary across input features
- Different emotion classes should produce visually distinct spike patterns

### 5.3 SNN Model (snn_model.py)

**Architecture:** 3-layer feedforward SNN with Leaky Integrate-and-Fire (LIF) neurons

```
Layer 1: Linear(310, 256) → snn.Leaky(beta, learn_beta=True)
Layer 2: Linear(256, 128) → snn.Leaky(beta, learn_beta=True)
Layer 3: Linear(128, 3)   → snn.Leaky(beta, learn_beta=True)
```

**Neuron model: snn.Leaky (LIF)**
- Membrane potential accumulates weighted input over time
- When membrane potential exceeds threshold → spike (binary 1)
- After spiking → membrane potential resets
- Between spikes → membrane potential decays by factor beta

**Parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| beta (decay rate) | ~0.85-0.9 initial | Learnable, one per layer |
| threshold | 1.0 | Fixed |
| learn_beta | True | Allows network to learn optimal decay |
| surrogate gradient | fast_sigmoid | For backprop through discontinuous spike function |
| reset mechanism | subtract | Subtract threshold from membrane potential after spike |

**Forward pass (per batch):**
1. Initialize membrane potentials and spike states to zero for all layers
2. Loop over T timesteps:
   a. Feed input spike train at timestep t through Layer 1
   b. Record spikes and membrane potentials at each layer
   c. Accumulate output layer spikes
3. Readout: sum of output spikes over all timesteps → (batch, 3)
4. Prediction: argmax of spike counts

**Outputs returned:**
- `spike_counts`: (batch, 3) — total output spikes per class per sample
- `membrane_traces`: (T, batch, 3) — output layer membrane potential over time
- `all_spikes`: dict of spike recordings per layer (for visualization)

### 5.4 Baseline MLP Model (baseline_mlp.py)

**Purpose:** Fair comparison baseline using the same parameter budget

**Architecture:**
```
Linear(310, 256) → ReLU → Dropout(0.3)
Linear(256, 128) → ReLU → Dropout(0.3)
Linear(128, 3)
```

**Design constraints:**
- Same layer dimensions as SNN (310 → 256 → 128 → 3)
- Comparable total parameter count
- Standard cross-entropy loss
- No spike encoding — takes raw features directly

### 5.5 Training (train.py)

**CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| --model | snn | "snn" or "baseline" |
| --epochs | 50 | Number of training epochs |
| --batch-size | 64 | Batch size |
| --lr | 1e-3 | Learning rate |
| --timesteps | 25 | SNN timesteps (ignored for baseline) |
| --use-synthetic | False | Use synthetic data instead of SEED |
| --data-path | data/ | Path to SEED data directory |
| --seed | 42 | Random seed for reproducibility |
| --save-dir | results/ | Directory for outputs |

**Training loop (SNN):**
1. Load data (real or synthetic)
2. Create train/test DataLoaders
3. For each epoch:
   a. For each batch:
      - Encode features → spike trains (T, batch, 310)
      - Forward pass through SNN
      - Compute cross-entropy loss on spike counts
      - Backprop with surrogate gradients
      - Update weights (Adam)
   b. Log: train loss, train accuracy, test loss, test accuracy
   c. Log (SNN only): sparsity %, total spike count
   d. Save best model checkpoint (by test accuracy)
4. Save training_curves.png (loss and accuracy over epochs)

**Training loop (Baseline MLP):**
- Same structure but no spike encoding step
- Raw features → model → cross-entropy → backprop
- Standard PyTorch training

**Output files:**
- `results/snn_checkpoint.pt` or `results/baseline_checkpoint.pt`
- `results/training_curves.png`
- Console: per-epoch metrics

### 5.6 Evaluation (evaluate.py)

**Inputs:** Trained model checkpoint + test data

**Outputs:**
- Classification report (precision, recall, F1 per class)
- Overall accuracy
- Confusion matrix PNG (`results/confusion_matrix_[model].png`)

**Process:**
1. Load best checkpoint
2. Run inference on full test set
3. Compute scikit-learn classification_report
4. Generate confusion matrix with matplotlib (dark theme)
5. Print and save all results

### 5.7 Visualization (visualize.py)

All visualizations use the consistent dark theme specified in Section 8.

**Visualization 1: Spike Raster per Emotion Class**
- THE key visual for content/reels
- 3-panel figure: one panel per emotion class (positive, neutral, negative)
- Each panel: neurons on Y-axis, timesteps on X-axis, dots where spikes occur
- Color-coded: green (positive), blue (neutral), red (negative)
- Shows how the SNN's internal activity differs across emotions
- Source: run a batch of samples from each class through the trained model, record hidden layer spikes

**Visualization 2: Accuracy Comparison Bar Chart**
- Side-by-side bars: SNN accuracy vs MLP accuracy
- Include error bars if running multiple seeds
- Annotate with exact percentages
- Title: "SNN vs Traditional MLP on EEG Emotion Classification"

**Visualization 3: Membrane Potential Traces**
- 3 subplots (one per output neuron / emotion class)
- X-axis: timesteps, Y-axis: membrane potential
- Show threshold line (dashed)
- Show spike events as vertical markers
- Overlay traces for different input emotion classes to show discrimination

**Visualization 4: Sparsity Over Training**
- X-axis: epoch, Y-axis: network sparsity (% of zero activations)
- Shows how the SNN maintains sparse activity through training
- Single line plot with confidence band if multiple runs

**Visualization 5 (optional): Spike Encoding Demo**
- 3-column figure showing raw DE features → normalized features → spike trains
- For one sample from each emotion class
- Demonstrates the encoding process visually

---

## 6. REPOSITORY STRUCTURE

```
neuroai-eeg-emotion/
├── README.md                   # Hero README with results, quick-start, architecture
├── requirements.txt            # Pinned dependencies
├── data/
│   ├── README.md               # SEED download instructions + expected structure
│   └── .gitkeep                # Placeholder so git tracks empty dir
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # SEED .mat loading + synthetic data generator
│   ├── spike_encoder.py        # DE features → spike trains (rate + delta coding)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── snn_model.py        # SNN with snnTorch Leaky neurons
│   │   └── baseline_mlp.py     # Baseline MLP for comparison
│   ├── train.py                # Training loop for both models
│   ├── evaluate.py             # Metrics, confusion matrix generation
│   └── visualize.py            # All visualization functions
├── notebooks/
│   └── demo.ipynb              # End-to-end walkthrough notebook
├── results/                    # Generated PNGs and checkpoints (gitignored except examples)
│   └── .gitkeep
├── LICENSE                     # MIT
└── .gitignore                  # Python, data files, checkpoints
```

---

## 7. DEPENDENCIES

### requirements.txt

```
torch>=2.0.0
snntorch>=0.9.0
scipy>=1.10.0
h5py>=3.8.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### System Requirements

- Python 3.9+
- CPU sufficient for development and synthetic data testing
- GPU (any CUDA-compatible) recommended for real data training
- ~2-3 GB disk for SEED DE features

---

## 8. VISUAL STYLE SPECIFICATION

All matplotlib visualizations must use this consistent theme:

| Element | Value |
|---------|-------|
| Background | #0f1419 |
| Figure facecolor | #0f1419 |
| Axes facecolor | #0f1419 |
| Green accent | #00e676 |
| Red accent | #ff5252 |
| Blue accent | #1da1f2 |
| Yellow/neutral accent | #ffab40 |
| Text / labels | #8899a6 |
| Panel borders | #2a3a4a |
| Box backgrounds | #1a2530 |
| Font family | sans-serif |
| Default font size | 14 |
| Title font size | 18 |
| DPI for saved PNGs | 180 |
| Spine color | #2a3a4a |
| Grid color | #1a2530 (subtle) |
| Tick color | #8899a6 |

**Implementation:** Create a `setup_dark_theme()` function in visualize.py that applies `plt.rcParams` updates globally. Call it before any plot generation.

---

## 9. README SPECIFICATION

The README.md must contain the following sections in order:

1. **Hero section:** Project name, one-liner thesis, hero image (spike raster visualization)
2. **Why SNNs for EEG?** — 3-4 sentences explaining the biological argument
3. **Key Results:** Accuracy comparison table/chart, spike raster preview
4. **Quick Start:** Clone → install → run on synthetic data (copy-paste terminal commands, under 5 commands)
5. **Architecture:** Diagram or description of the pipeline (data → encoding → SNN → classification)
6. **Dataset:** SEED description, download instructions (link to data/README.md), citation
7. **Visualizations:** Embedded PNGs of key outputs with captions
8. **Training:** How to train on real data, expected training time, hyperparameter tuning tips
9. **Project Structure:** Directory tree
10. **References:** SEED paper, combra-lab paper, snnTorch
11. **Author:** Tejas J Bharadwaj, CMU SCS — with links to LinkedIn, Instagram, personal website
12. **License:** MIT badge + link

---

## 10. data/README.md SPECIFICATION

Must contain:
1. Dataset name and source (SEED, SJTU BCMI Lab)
2. How to request access (link to BCMI Lab website, use academic email)
3. What to download (ExtractedFeatures folder + label.mat)
4. Where to place files (expected directory structure)
5. Citation in BibTeX format
6. Note that the project can run on synthetic data without downloading SEED

---

## 11. CONSTRAINTS & RULES

1. **NEVER mention** Presonance, TRIBE v2, fMRI brain decoding, or neuromarketing in any file (README, comments, code, notebooks)
2. The repo is purely about SNN + EEG emotion classification — brain-inspired AI for brain data
3. All visualizations must use the dark theme from Section 8 for brand consistency
4. The synthetic data fallback must be fully functional — the entire pipeline must work without SEED data
5. README must include Tejas's name, CMU affiliation, and links to social profiles
6. MIT license for maximum openness
7. Code must be clean, well-commented, and follow PEP 8 style
8. All CLI arguments must have sensible defaults so commands work without flags
9. The notebook (demo.ipynb) must be runnable end-to-end on synthetic data

---

## 12. POST-BUILD DEPLOYMENT PLAN

After the pipeline is built and tested on synthetic data:

1. Download SEED ExtractedFeatures to GCS bucket
2. Spin up GCP VM with GPU, pull data from bucket
3. Train SNN and MLP on real data (50-100 epochs, ~30 min on T4/L4)
4. Download checkpoint + regenerated visualizations
5. Update README with real-data results
6. Push to GitHub, pin on profile
7. Post to communities:
   - r/MachineLearning ("I built X" post)
   - r/neuroscience
   - Open Neuromorphic Discord
   - snnTorch GitHub Discussions
   - Hacker News (Show HN)
8. Open PR/issue on snnTorch tutorials repo suggesting EEG emotion tutorial
9. Film Reel #2 using real-data visualizations as B-roll

---

## 13. KNOWN FAILURE MODES & MITIGATIONS

| Problem | Cause | Fix |
|---------|-------|-----|
| scipy.io.loadmat fails | SEED .mat files may be MATLAB v7.3 (HDF5) | Fall back to h5py; data_loader must try both |
| SNN loss doesn't decrease | LR too high or spike encoding producing trivial output | Reduce LR to 1e-4, increase timesteps to 50, verify encoder output |
| SNN produces zero spikes | Input values too small or threshold too high | Increase input scaling in encoder, reduce threshold below 1.0 |
| SNN produces all spikes (no sparsity) | Beta too low (too much memory) or threshold too low | Increase beta toward 0.95, increase threshold |
| ~33% accuracy on synthetic data | Random data, 3 classes, expected behavior | Not a bug — pipeline is working correctly |
| SEED .mat key structure unexpected | Different versions may use different key names | Print all keys with loadmat, inspect shapes, adapt data_loader dynamically |
| Visualizations blank/empty | Model didn't produce spikes during inference | Print spike counts during inference; check model is in eval mode |
| OOM on GPU | Batch too large for GPU memory | Reduce batch size to 32 or 16 |
| Import errors on Mac (MPS) | snnTorch may have MPS compatibility issues | Force CPU with --device cpu flag |
