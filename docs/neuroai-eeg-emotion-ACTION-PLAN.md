# ACTION PLAN: neuroai-eeg-emotion
## Granular Build Plan for Claude Code Execution

---

## HOW TO USE THIS DOCUMENT

This is a step-by-step build plan for Claude Code. Execute each phase sequentially. At the end of each phase, there is a **CHECKPOINT** — a set of concrete deliverables you must produce and show before moving to the next phase. Do not skip phases. Do not move forward until the checkpoint is satisfied.

**Companion document:** Refer to `neuroai-eeg-emotion-PRD.md` for all specifications, architecture details, visual style, and constraints.

**Environment:** macOS local development. Training on real data will happen separately on GCP.

---

## PHASE 1: PROJECT SCAFFOLDING
**Estimated time: 15-20 min**

### Step 1.1: Create Directory Structure

Create the full project directory tree:

```
neuroai-eeg-emotion/
├── data/
├── src/
│   ├── models/
├── notebooks/
├── results/
```

Create all directories including nested ones. Ensure `data/`, `results/` have `.gitkeep` files so git tracks them.

### Step 1.2: Create requirements.txt

Create `requirements.txt` with these dependencies:

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

### Step 1.3: Create Python Package Files

Create `__init__.py` files:
- `src/__init__.py` — empty
- `src/models/__init__.py` — empty

### Step 1.4: Create Virtual Environment and Install Dependencies

1. Create a Python virtual environment in the project root: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install all requirements: `pip install -r requirements.txt`
4. Verify installation succeeded with zero errors

### Step 1.5: Verify All Imports

Create and run a temporary test script that imports every dependency:

```
import torch
import snntorch as snn
import snntorch.spikegen as spikegen
import snntorch.surrogate as surrogate
import scipy.io
import h5py
import numpy as np
import sklearn
import matplotlib
import tqdm
```

Print the version of each package. Verify no import errors.

### Step 1.6: Create .gitignore

Create `.gitignore` with standard Python ignores plus:
- `venv/`
- `data/*.mat`
- `data/SEED*/`
- `data/ExtractedFeatures/`
- `results/*.pt` (checkpoints)
- `__pycache__/`
- `.DS_Store`
- `*.pyc`
- `.ipynb_checkpoints/`

Do NOT ignore `results/*.png` — we want example visualizations tracked.

### Step 1.7: Create LICENSE

Create MIT license file with copyright holder: Tejas J Bharadwaj

### Step 1.8: Initialize Git

1. `git init`
2. `git add -A`
3. `git commit -m "Phase 1: Project scaffolding"`

### CHECKPOINT — Phase 1 Deliverables:

Show the following:
1. Full directory tree output (`find . -not -path './venv/*' -not -path './.git/*' | sort`)
2. Output of the import verification script showing all package versions
3. Contents of requirements.txt
4. Contents of .gitignore
5. Git log showing the initial commit

---

## PHASE 2: DATA LOADING PIPELINE
**Estimated time: 30-40 min**

### Step 2.1: Build Synthetic Data Generator

In `src/data_loader.py`, implement a function `generate_synthetic_data()`:

**Inputs:**
- `n_samples` (int, default 3000): total number of samples
- `n_features` (int, default 310): feature dimension (62 channels × 5 frequency bands)
- `n_classes` (int, default 3): number of emotion classes
- `seed` (int, default 42): random seed

**Behavior:**
- Generate `n_samples` random feature vectors, uniformly distributed in [0, 1]
- Generate balanced labels: exactly `n_samples // n_classes` samples per class
- Shuffle the data
- Return X as numpy array (n_samples, n_features) and y as numpy array (n_samples,) with values in {0, 1, 2}

**Design notes:**
- This must produce the EXACT same interface as the real data loader
- Include a docstring explaining this is for pipeline testing and ~33% accuracy is expected

### Step 2.2: Build SEED Data Loader

In the same `src/data_loader.py`, implement a function `load_seed_data()`:

**Inputs:**
- `data_path` (str): path to the directory containing SEED .mat files (ExtractedFeatures folder)
- `label_path` (str, optional): path to label.mat if separate

**Behavior:**

1. **Discover .mat files:** Scan `data_path` for all .mat files. Sort them for deterministic ordering.

2. **Load each .mat file:** 
   - First try `scipy.io.loadmat(filepath)` 
   - If that fails (MATLAB v7.3), fall back to `h5py.File(filepath, 'r')`
   - Print the keys found in the first file for debugging

3. **Extract DE features per trial:**
   - Each .mat file contains one session (15 trials for 15 film clips)
   - The keys for each trial may be named like `de_LDS1`, `de_LDS2`, ..., `de_LDS15` or similar
   - Each trial's data has shape approximately (N_windows, 62, 5) or (N_windows, 310)
   - If shape is (N_windows, 62, 5): reshape to (N_windows, 310) by flattening last two dims
   - If shape is already (N_windows, 310): use as-is
   - **CRITICAL: The exact key names and shapes MUST be inspected at runtime.** Print the keys and the shape of the first value found. Adapt accordingly. Do not hardcode assumptions about key names.

4. **Load labels:**
   - Look for a `label.mat` file in the data directory or parent directory
   - The label array should have 15 values per session (one per film clip)
   - Labels are -1, 0, +1 → map to 0, 1, 2
   - If labels are embedded in session files under a key like `label`, extract from there

5. **Assign labels to windows:**
   - Each trial has N_windows of DE features
   - All windows in a trial get the same emotion label (the label for that film clip)

6. **Aggregate:**
   - Stack all windows from all trials from all sessions
   - Output X (total_samples, 310) and y (total_samples,) 

7. **Print summary:**
   - Total samples
   - Samples per class
   - Feature shape
   - Number of sessions loaded

### Step 2.3: Build Unified Data Interface

Implement a function `load_data()`:

**Inputs:**
- `use_synthetic` (bool, default False)
- `data_path` (str, default "data/")
- `test_size` (float, default 0.2)
- `seed` (int, default 42)

**Behavior:**
1. If `use_synthetic`: call `generate_synthetic_data()`
2. Else: call `load_seed_data(data_path)`
3. Perform stratified train/test split using sklearn's `train_test_split`
4. Convert to PyTorch tensors (float32 for X, long for y)
5. Return: `X_train, X_test, y_train, y_test`

### Step 2.4: Test Data Loading

Run data loader with synthetic data. Print and verify:
- X_train shape, X_test shape
- y_train shape, y_test shape
- Label distribution in train and test sets
- Value range of X (should be [0, 1])
- Data types of all tensors

### Step 2.5: Git Commit

`git add -A && git commit -m "Phase 2: Data loading pipeline with synthetic fallback"`

### CHECKPOINT — Phase 2 Deliverables:

Show the following:
1. Terminal output of running data_loader.py with synthetic data showing: shapes, label distribution, value ranges
2. Confirmation that X shape is (N, 310) and y values are in {0, 1, 2}
3. Train/test split sizes (should be ~80/20)

---

## PHASE 3: SPIKE ENCODING
**Estimated time: 20-30 min**

### Step 3.1: Build Rate Encoder

In `src/spike_encoder.py`, implement:

**Function `encode_rate()`:**

**Inputs:**
- `X` (torch.Tensor): feature tensor of shape (batch, 310)
- `T` (int, default 25): number of timesteps
- `gain` (float, default 1.0): scaling factor for spike probability

**Behavior:**
1. Clamp X to [0, 1] range (safety)
2. Use `snntorch.spikegen.rate(X, num_steps=T, gain=gain)`
3. Output shape: (T, batch, 310) — time-first format as required by snnTorch
4. Output is binary (0s and 1s)

### Step 3.2: Build Delta Encoder (Secondary)

**Function `encode_delta()`:**

**Inputs:**
- `X` (torch.Tensor): feature tensor of shape (batch, 310)
- `T` (int, default 25): number of timesteps
- `threshold` (float, default 0.1): change threshold for spike generation

**Behavior:**
1. Create a time-varying version of X by adding small random noise per timestep
2. Compute temporal differences
3. Generate spikes where the absolute difference exceeds threshold
4. Output shape: (T, batch, 310)

### Step 3.3: Build Unified Encoding Interface

**Function `encode_spikes()`:**

**Inputs:**
- `X` (torch.Tensor): features
- `method` (str, default "rate"): "rate" or "delta"
- `T` (int, default 25): timesteps
- Additional kwargs passed to the specific encoder

**Returns:** spike tensor of shape (T, batch, 310)

### Step 3.4: Build Normalization Function

**Function `normalize_features()`:**

**Inputs:**
- `X_train` (torch.Tensor): training features
- `X_test` (torch.Tensor): test features

**Behavior:**
1. Compute min and max per feature from X_train ONLY (prevent data leakage)
2. Normalize X_train and X_test to [0, 1] using these statistics
3. Clamp to [0, 1] to handle any test samples outside training range
4. Return normalized X_train, normalized X_test, and the (min, max) statistics for later use

### Step 3.5: Test Spike Encoding

Run encoding on a small batch of synthetic data. Print and verify:
- Input shape: (batch, 310)
- Output shape: (T, batch, 310)
- Sparsity percentage: `(1 - spike_tensor.mean()) * 100`
- Verify sparsity is between 40-95%
- Verify output is binary (only 0s and 1s)

### Step 3.6: Generate Spike Encoding Demo Visualization

Create a visualization showing spike encoding for 3 sample inputs (one from each emotion class, using synthetic data):
- 3-column figure
- Each column: spike raster for one sample (features on Y-axis, timesteps on X-axis)
- Column titles: "Positive", "Neutral", "Negative"
- Use dark theme from PRD Section 8
- Save to `results/spike_encoding_demo.png`

### Step 3.7: Git Commit

`git add -A && git commit -m "Phase 3: Spike encoding (rate + delta)"`

### CHECKPOINT — Phase 3 Deliverables:

Show the following:
1. Terminal output: input shape, output shape, sparsity percentage, binary verification
2. `results/spike_encoding_demo.png` — the 3-column spike raster visualization
3. Confirmation that encode_rate and encode_delta both produce correct output shapes

---

## PHASE 4: MODEL ARCHITECTURE
**Estimated time: 25-35 min**

### Step 4.1: Build SNN Model

In `src/models/snn_model.py`, implement class `SpikingNN(nn.Module)`:

**Constructor (`__init__`):**
- Accepts: `input_size` (default 310), `hidden1` (default 256), `hidden2` (default 128), `output_size` (default 3), `beta` (default 0.9), `timesteps` (default 25)
- Create 3 Linear layers: input→hidden1, hidden1→hidden2, hidden2→output
- Create 3 snn.Leaky neurons with `beta=beta, learn_beta=True, threshold=1.0`
- Set surrogate gradient: `spike_grad = surrogate.fast_sigmoid(slope=25)`
- Pass `spike_grad` to each Leaky neuron
- Store timesteps as an attribute

**Forward method:**
- Input: `x` of shape (T, batch, input_size) — pre-encoded spike trains
- Initialize membrane potential and spike state for each layer using `.init_leaky()`
- Create storage lists for: output spike recording, output membrane recording, hidden layer spike recordings (for visualization)
- Loop over T timesteps:
  - Pass `x[t]` through fc1 → leaky1 → get (spike1, mem1)
  - Pass spike1 through fc2 → leaky2 → get (spike2, mem2)
  - Pass spike2 through fc3 → leaky3 → get (spike3, mem3)
  - Append spike3 to output spike list
  - Append mem3 to output membrane list
  - Append spike1, spike2 to hidden spike lists
- Stack output spikes: (T, batch, output_size)
- Compute spike counts: sum over T → (batch, output_size)
- Stack membrane traces: (T, batch, output_size)
- Return dict with keys: `spike_counts`, `membrane_traces`, `output_spikes`, `hidden1_spikes`, `hidden2_spikes`

**Method `count_parameters()`:**
- Return total number of trainable parameters

### Step 4.2: Build Baseline MLP

In `src/models/baseline_mlp.py`, implement class `BaselineMLP(nn.Module)`:

**Constructor:**
- Accepts: `input_size` (default 310), `hidden1` (default 256), `hidden2` (default 128), `output_size` (default 3), `dropout` (default 0.3)
- Create: fc1(input→hidden1), relu1, dropout1, fc2(hidden1→hidden2), relu2, dropout2, fc3(hidden2→output)

**Forward method:**
- Input: `x` of shape (batch, input_size) — raw features, NOT spike trains
- Pass through all layers sequentially
- Return logits of shape (batch, output_size)

**Method `count_parameters()`:**
- Return total number of trainable parameters

### Step 4.3: Test Both Models

Write and run a test that:

1. Creates a random input tensor:
   - For SNN: shape (25, 8, 310) — T=25, batch=8, features=310
   - For MLP: shape (8, 310) — batch=8, features=310

2. Instantiates both models

3. Runs forward pass on both

4. Prints:
   - SNN output spike_counts shape (should be [8, 3])
   - SNN membrane_traces shape (should be [25, 8, 3])
   - SNN hidden1_spikes shape
   - SNN hidden2_spikes shape
   - MLP output shape (should be [8, 3])
   - SNN parameter count
   - MLP parameter count
   - Both should have similar parameter counts

5. Verify no errors during forward pass

### Step 4.4: Git Commit

`git add -A && git commit -m "Phase 4: SNN and baseline MLP model architectures"`

### CHECKPOINT — Phase 4 Deliverables:

Show the following:
1. Terminal output of model test: all shapes, parameter counts for both models
2. Confirmation that SNN forward pass returns all expected keys (spike_counts, membrane_traces, output_spikes, hidden1_spikes, hidden2_spikes)
3. Confirmation parameter counts are in the same ballpark

---

## PHASE 5: TRAINING LOOP
**Estimated time: 35-45 min**

### Step 5.1: Build Training Script

In `src/train.py`, implement the full training pipeline:

**CLI Argument Parsing:**
- Use argparse with all arguments from PRD Section 5.5
- Every argument must have a default value so the script runs with zero flags
- Add `--device` argument (default: auto-detect cuda > mps > cpu)

**Function `train_snn()`:**

Inputs: model, train_loader, test_loader, optimizer, criterion, encoder, device, epochs, timesteps, save_dir

Behavior per epoch:
1. Set model to train mode
2. For each batch in train_loader:
   a. Move data to device
   b. Encode features → spike trains using the encoder
   c. Forward pass through SNN → get output dict
   d. Compute loss: `nn.CrossEntropyLoss()` on `output['spike_counts']` vs labels
   e. Backward pass
   f. Optimizer step
   g. Track: running loss, correct predictions, total spikes, total possible spikes
3. Compute epoch train metrics: loss, accuracy, sparsity
4. Evaluate on test set (no gradients):
   a. Same encoding + forward pass
   b. Compute test loss, test accuracy
5. Print epoch summary: epoch, train_loss, train_acc, test_loss, test_acc, sparsity%
6. If test_acc is best so far: save checkpoint (model state dict, optimizer state dict, epoch, test_acc)
7. Append all metrics to history lists

Return: training history dict

**Function `train_baseline()`:**

Same structure as train_snn but:
- No spike encoding step
- Raw features go directly into model
- No sparsity tracking
- Loss computed on model output logits

**Function `plot_training_curves()`:**

Inputs: snn_history (optional), baseline_history (optional), save_path

Behavior:
1. Create 2-row figure: top row = loss curves, bottom row = accuracy curves
2. If both histories provided: plot both on same axes with different colors
3. SNN in green (#00e676), baseline in blue (#1da1f2)
4. Apply dark theme from PRD Section 8
5. Save to `results/training_curves.png`

**Main block:**
1. Parse args
2. Load data (synthetic or SEED)
3. Normalize features
4. Create DataLoaders (train + test)
5. Instantiate model (SNN or baseline based on --model flag)
6. Create optimizer (Adam)
7. Create loss function (CrossEntropyLoss)
8. Call appropriate train function
9. Plot training curves
10. Print final best test accuracy

### Step 5.2: Test SNN Training on Synthetic Data

Run: `python src/train.py --model snn --epochs 5 --use-synthetic --batch-size 64`

Verify:
- Training runs without errors for all 5 epochs
- Loss and accuracy are printed each epoch
- Sparsity percentage is printed each epoch
- Checkpoint is saved to results/
- training_curves.png is generated
- Accuracy is approximately 33% (expected for random data)

### Step 5.3: Test Baseline Training on Synthetic Data

Run: `python src/train.py --model baseline --epochs 5 --use-synthetic --batch-size 64`

Verify:
- Training runs without errors
- Loss decreases (it may still hover around random chance on synthetic data)
- Checkpoint is saved
- training_curves.png is updated/generated

### Step 5.4: Git Commit

`git add -A && git commit -m "Phase 5: Training loop for SNN and baseline MLP"`

### CHECKPOINT — Phase 5 Deliverables:

Show the following:
1. Full terminal output of SNN training (5 epochs, synthetic data) — showing per-epoch loss, accuracy, sparsity
2. Full terminal output of baseline training (5 epochs, synthetic data)
3. `results/training_curves.png`
4. Confirmation that checkpoint files exist in results/
5. File sizes of checkpoints

---

## PHASE 6: EVALUATION & VISUALIZATION
**Estimated time: 35-45 min**

### Step 6.1: Build Evaluation Script

In `src/evaluate.py`, implement:

**Function `evaluate_model()`:**

Inputs: model, test_loader, device, model_type ("snn" or "baseline"), encoder (for SNN), timesteps

Behavior:
1. Set model to eval mode
2. Run inference on full test set (no gradients)
3. Collect all predictions and true labels
4. Compute:
   - sklearn classification_report (precision, recall, F1 per class)
   - Confusion matrix
   - Overall accuracy
5. Return: predictions, true labels, classification report string, confusion matrix array

**Function `plot_confusion_matrix()`:**

Inputs: confusion_matrix, class_names (["Negative", "Neutral", "Positive"]), save_path, model_name

Behavior:
1. Create figure with dark theme
2. Plot confusion matrix as a heatmap
3. Annotate cells with counts
4. X-axis: Predicted, Y-axis: True
5. Title: f"Confusion Matrix — {model_name}"
6. Color map: custom from #0f1419 (dark) to #00e676 (green)
7. Save to save_path

**CLI interface:**
- `--model`: snn or baseline
- `--checkpoint`: path to checkpoint file
- `--use-synthetic`: flag
- `--data-path`: path to data
- `--timesteps`: for SNN encoding
- `--device`: auto-detect

### Step 6.2: Build Visualization Module

In `src/visualize.py`, implement:

**Function `setup_dark_theme()`:**
- Apply all plt.rcParams from PRD Section 8
- Call this before any plot generation

**Function `plot_spike_rasters()`:**

Inputs: model, test_loader, encoder, device, timesteps, save_path, n_samples (default 5)

Behavior:
1. Select `n_samples` samples from each emotion class
2. Run through model, collect hidden layer spikes (hidden1_spikes)
3. Create 3-panel figure (1 row, 3 columns)
4. Each panel: spike raster for that emotion class
   - Y-axis: neuron index (show first 50 neurons for clarity)
   - X-axis: timestep
   - Plot dots/markers where spikes occur
   - Panel title: emotion class name
   - Panel color accent: Negative=#ff5252, Neutral=#1da1f2, Positive=#00e676
5. Overall title: "SNN Spike Activity by Emotion Class"
6. Save to save_path

**Function `plot_accuracy_comparison()`:**

Inputs: snn_accuracy, baseline_accuracy, save_path

Behavior:
1. Simple bar chart: 2 bars
2. SNN bar in green (#00e676), MLP bar in blue (#1da1f2)
3. Annotate exact percentages on top of bars
4. Title: "Emotion Classification Accuracy: SNN vs MLP"
5. Y-axis: "Accuracy (%)"
6. Dark theme
7. Save to save_path

**Function `plot_membrane_traces()`:**

Inputs: model, sample_batch, encoder, device, timesteps, save_path

Behavior:
1. Take one sample from each emotion class
2. Run through model, get membrane_traces (T, 1, 3)
3. Create 3-subplot figure (3 rows or 3 columns)
4. Each subplot: one output neuron's membrane potential over time
5. Plot threshold line (dashed white at y=1.0)
6. Mark spike events with vertical lines or markers
7. Overlay traces from all 3 emotion inputs in different colors
8. Subplot titles: "Output Neuron 0 (Negative)", etc.
9. Save to save_path

**Function `plot_sparsity_over_training()`:**

Inputs: training_history (dict with 'sparsity' key), save_path

Behavior:
1. Line plot: x=epoch, y=sparsity percentage
2. Green line (#00e676)
3. Fill between for visual weight
4. Title: "Network Sparsity During Training"
5. Save to save_path

### Step 6.3: Run Evaluation on Synthetic-Trained Models

1. Run evaluate.py on the SNN checkpoint from Phase 5
2. Run evaluate.py on the baseline checkpoint from Phase 5
3. Generate all visualizations:
   - Confusion matrix for SNN
   - Confusion matrix for baseline
   - Spike raster per emotion class
   - Accuracy comparison
   - Membrane potential traces
   - Sparsity over training

### Step 6.4: Verify All Visualization Files

List all PNG files in results/. Verify each one exists and is non-zero size:
- `results/confusion_matrix_snn.png`
- `results/confusion_matrix_baseline.png`
- `results/spike_rasters.png`
- `results/accuracy_comparison.png`
- `results/membrane_traces.png`
- `results/sparsity_training.png`
- `results/spike_encoding_demo.png` (from Phase 3)
- `results/training_curves.png` (from Phase 5)

### Step 6.5: Git Commit

`git add -A && git commit -m "Phase 6: Evaluation metrics and visualizations"`

### CHECKPOINT — Phase 6 Deliverables:

Show the following:
1. Classification report for SNN (printed text)
2. Classification report for baseline (printed text)
3. All PNG files listed with file sizes
4. Display `results/spike_rasters.png`
5. Display `results/accuracy_comparison.png`
6. Display `results/confusion_matrix_snn.png`
7. Display `results/membrane_traces.png`

---

## PHASE 7: README & DOCUMENTATION
**Estimated time: 25-35 min**

### Step 7.1: Create Main README.md

Follow the specification in PRD Section 9. Create `README.md` in project root.

**Content requirements:**

**Hero section:**
```markdown
# neuroai-eeg-emotion 🧠⚡

**Emotion classification from real EEG brain data using Spiking Neural Networks.**

Brain data IS spikes. Why would you use a non-spiking model to process it?

![Spike Rasters](results/spike_rasters.png)
```

**"Why SNNs for EEG?" section:**
- Explain that biological neurons communicate through discrete spikes
- EEG captures aggregate neural spike activity
- SNNs (Leaky Integrate-and-Fire neurons) mirror this: accumulate → threshold → fire → reset
- Result: more biologically faithful, more energy-efficient (cite combra-lab 95% figure), interpretable spike dynamics

**"Key Results" section:**
- Embed accuracy comparison chart
- Table of SNN vs MLP accuracy, parameter count, relative compute
- Note: results shown are on synthetic data / SEED data (update after real training)

**"Quick Start" section:**
```bash
git clone https://github.com/[handle]/neuroai-eeg-emotion.git
cd neuroai-eeg-emotion
pip install -r requirements.txt
python src/train.py --model snn --epochs 10 --use-synthetic
python src/evaluate.py --model snn --checkpoint results/snn_checkpoint.pt --use-synthetic
```
- Must work copy-paste with zero modifications

**"Architecture" section:**
- Text description of the pipeline: Data → Spike Encoding → SNN → Classification
- Layer specs: 310 → 256 → 128 → 3
- Key design decisions: Leaky neurons, learnable beta, rate coding, surrogate gradients

**"Dataset" section:**
- SEED dataset description
- Link to data/README.md for download instructions
- Citation in proper academic format

**"Visualizations" section:**
- Embed all key PNGs with captions:
  - Spike rasters showing per-emotion neural activity patterns
  - Membrane potential traces showing how output neurons discriminate emotions
  - Training curves
  - Confusion matrix

**"Training on Real Data" section:**
- How to download and place SEED data
- Command to train on real data: `python src/train.py --model snn --epochs 50 --data-path data/ExtractedFeatures/`
- Expected training time on GPU
- Tips: learning rate, timesteps, batch size

**"Project Structure" section:**
- Full directory tree

**"References" section:**
- SEED paper (Zheng & Lu, 2015)
- combra-lab TMLR paper
- snnTorch (Eshraghian et al.)

**"Author" section:**
```markdown
**Tejas J Bharadwaj**
CMU School of Computer Science, Class of 2030

- [LinkedIn](linkedin.com/in/tejas-bharadwaj)
- [Instagram](instagram.com/[handle])
- [Website](tejasbharadwaj.com)
```

**"License" section:**
- MIT badge
- Link to LICENSE file

### Step 7.2: Create data/README.md

Follow PRD Section 10 specification:

1. Title: "SEED Dataset — Download Instructions"
2. Source: SJTU BCMI Lab
3. How to request access: visit https://bcmi.sjtu.edu.cn/home/seed/ and request with academic email
4. What to download: `ExtractedFeatures` folder from SEED_EEG
5. Where to place: `data/ExtractedFeatures/`
6. Expected structure after placement
7. BibTeX citation for SEED
8. Note: project runs on synthetic data without SEED download

### Step 7.3: Review and Polish

1. Read through entire README for clarity and flow
2. Verify all image paths are correct
3. Verify quick-start commands actually work
4. Check for any mention of Presonance, TRIBE v2, fMRI, or neuromarketing — MUST NOT EXIST
5. Verify author section has correct name and CMU affiliation

### Step 7.4: Git Commit

`git add -A && git commit -m "Phase 7: README and documentation"`

### CHECKPOINT — Phase 7 Deliverables:

Show the following:
1. Full contents of README.md
2. Full contents of data/README.md
3. Confirmation that no prohibited terms (Presonance, TRIBE, fMRI, neuromarketing) appear anywhere in the repo: `grep -r "presonance\|TRIBE\|fMRI\|neuromarketing" --include="*.py" --include="*.md" .`

---

## PHASE 8: DEMO NOTEBOOK & FINAL INTEGRATION TEST
**Estimated time: 25-35 min**

### Step 8.1: Create Demo Notebook

Create `notebooks/demo.ipynb` — a Jupyter notebook that walks through the entire pipeline end-to-end.

**Cell structure:**

Cell 1: Title + Introduction
- Markdown explaining the project, the thesis, what the notebook demonstrates

Cell 2: Imports
- Import all necessary modules from src/

Cell 3: Load Data
- Load synthetic data using data_loader
- Print shapes and label distribution

Cell 4: Spike Encoding
- Encode a small batch
- Print shapes and sparsity
- Generate inline spike encoding visualization

Cell 5: Create Models
- Instantiate SNN and MLP
- Print parameter counts

Cell 6: Quick Training Demo
- Train SNN for 5 epochs on synthetic data
- Print per-epoch metrics

Cell 7: Evaluation
- Run evaluation on trained SNN
- Print classification report
- Generate confusion matrix inline

Cell 8: Visualizations
- Generate spike rasters
- Generate membrane traces
- All displayed inline

Cell 9: Conclusion
- Summary of what was demonstrated
- Link to README for real data training instructions

**Important:** The notebook must be runnable from top to bottom on synthetic data with no modifications. All cells must execute without errors.

### Step 8.2: Run Full Integration Test

Execute the complete pipeline from scratch as if you just cloned the repo:

1. Verify clean state: remove any existing results files
2. Run: `python src/train.py --model snn --epochs 5 --use-synthetic`
3. Run: `python src/train.py --model baseline --epochs 5 --use-synthetic`
4. Run: `python src/evaluate.py --model snn --checkpoint results/snn_checkpoint.pt --use-synthetic`
5. Run: `python src/evaluate.py --model baseline --checkpoint results/baseline_checkpoint.pt --use-synthetic`
6. Run a visualization generation command that produces all plots
7. Verify all expected files exist in results/

### Step 8.3: Lint All Python Files

Run flake8 or similar linter on all .py files in src/:
- `flake8 src/ --max-line-length=120 --ignore=E501,W503`
- Fix any critical issues (undefined variables, syntax errors)
- Minor style warnings are acceptable

### Step 8.4: Final Git Commit

```bash
git add -A
git commit -m "Phase 8: Demo notebook and final integration test"
```

### Step 8.5: Generate Final Summary

Print the following final summary:

1. Full directory tree of the project
2. `git log --oneline` showing all phase commits
3. List all files in results/ with sizes
4. Total line count of all Python files: `find src/ -name "*.py" | xargs wc -l`
5. Contents of requirements.txt

### CHECKPOINT — Phase 8 Deliverables:

Show the following:
1. Full terminal output of the integration test (Steps 8.2.2 through 8.2.6)
2. List of all files in results/ with sizes
3. Git log showing all 8 phase commits
4. Full directory tree
5. Lint output showing no critical errors
6. Confirmation: "Pipeline is complete. Ready for real data training on GCP."

---

## POST-BUILD: WHAT TEJAS DOES NEXT

This section is NOT executed by Claude Code. This is a reference for Tejas.

### Immediate (Day 1 after build):
1. Upload SEED ExtractedFeatures to GCS bucket
2. Spin up GCP VM with T4/L4 GPU
3. Clone repo to VM, pull data from bucket
4. Train SNN: `python src/train.py --model snn --epochs 50 --data-path data/ExtractedFeatures/`
5. Train baseline: `python src/train.py --model baseline --epochs 50 --data-path data/ExtractedFeatures/`
6. Generate all visualizations with real-data-trained models
7. Download results/ folder back to Mac
8. Update README with real-data results and screenshots
9. Push to GitHub, pin on profile

### Community posting (Day 2):
1. r/MachineLearning — "I built X" format post
2. r/neuroscience — cross-post with neuroscience framing
3. Open Neuromorphic Discord — share in #projects
4. snnTorch GitHub Discussions — share as a showcase
5. Hacker News — "Show HN: Emotion classification from EEG using Spiking Neural Networks"

### snnTorch contribution (Day 3):
1. Open issue on snnTorch tutorials repo
2. Propose "EEG Emotion Classification with snnTorch" tutorial
3. Reference your repo as the working implementation
4. If accepted, write and submit the tutorial PR

### Content (Day 4):
1. Film Reel #2 using real-data spike raster visualizations as B-roll
2. Post LinkedIn article about the project
3. Draft Medium article if time permits

---

## FAILURE MODES & RECOVERY

If any phase fails, refer to this table before moving forward:

| Failure | Recovery |
|---------|----------|
| pip install snntorch fails | Try `pip install snntorch==0.9.1` specifically; if torch version conflict, install torch first then snntorch |
| scipy.io.loadmat fails on SEED data | Switch to h5py: `h5py.File(path, 'r')` — SEED files may be MATLAB v7.3 HDF5 format |
| SNN forward pass errors | Check input shape is (T, batch, 310) not (batch, T, 310); snnTorch expects time-first |
| SNN loss is NaN | Reduce learning rate to 1e-4; check for division by zero in spike count readout |
| SNN produces zero spikes (all outputs are 0) | Increase gain in spike encoder; reduce threshold from 1.0 to 0.5; verify encoded spikes are non-trivial (print mean) |
| SNN produces all spikes (0% sparsity) | Increase beta toward 0.95; increase threshold to 1.5 |
| ~33% accuracy on synthetic data | CORRECT BEHAVIOR — random data with 3 classes. Not a bug. |
| Matplotlib figures are blank | Ensure setup_dark_theme() is called; check that data passed to plot functions is non-empty |
| Out of memory | Reduce batch size to 32 or 16; if on CPU, reduce timesteps to 10 for testing |
| Git commit fails | Ensure git is initialized and you're in the project root |
| Notebook won't run | Verify the working directory and sys.path includes the project root so src/ imports work |
| Import errors "no module named src" | Run scripts from project root; or add `sys.path.insert(0, '..')` in notebook |
