# neuroai-eeg-emotion

Emotion classification from EEG brain data using Spiking Neural Networks (snnTorch). Brain data IS spikes — SNNs are the natural model for it.

## Tech Stack

- Python 3.9+
- PyTorch >= 2.0, snnTorch >= 0.9
- scipy, h5py (MATLAB .mat loading)
- scikit-learn (metrics), matplotlib (viz), tqdm
- No frontend, no web, no deployment — pure Python ML pipeline

## Project Structure

```
src/
├── data_loader.py       # SEED .mat loading + synthetic data fallback
├── spike_encoder.py     # DE features → spike trains (rate/delta coding)
├── models/
│   ├── snn_model.py     # 3-layer SNN with snn.Leaky neurons
│   └── baseline_mlp.py  # Comparison MLP, same param budget
├── train.py             # Training loop with CLI args
├── evaluate.py          # Metrics + confusion matrix
└── visualize.py         # Spike rasters, membrane traces, comparisons
```

## Build Plan

- Read `docs/ACTION-PLAN.md` for the full phase-by-phase build plan
- Read `docs/PRD.md` for all specifications, architecture, hyperparameters, visual style
- Execute phases 1-8 in order. Each phase has a CHECKPOINT — show deliverables before proceeding.

## Code Style

- PEP 8, max line length 120
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Imports grouped: stdlib, third-party, local (blank line between groups)

## Commands

- Install: `pip install -r requirements.txt`
- Train SNN: `python src/train.py --model snn --epochs 5 --use-synthetic`
- Train MLP: `python src/train.py --model baseline --epochs 5 --use-synthetic`
- Evaluate: `python src/evaluate.py --model snn --checkpoint results/snn_checkpoint.pt --use-synthetic`
- Lint: `flake8 src/ --max-line-length=120 --ignore=E501,W503`

## Visual Style (all matplotlib plots)

- Background: #0f1419, text: #8899a6, borders: #2a3a4a
- Green: #00e676, Red: #ff5252, Blue: #1da1f2, Yellow: #ffab40
- Font: sans-serif 14pt, DPI: 180
- Create `setup_dark_theme()` in visualize.py — call before any plot

## Git Workflow

- One commit per phase: `"Phase N: <description>"`
- Conventional commit messages
- Never commit venv/, __pycache__/, .mat data files, or .pt checkpoints
- DO commit example result PNGs to results/

## Prohibited

- NEVER mention: Presonance, TRIBE v2, fMRI brain decoding, neuromarketing
- No hardcoded file paths — always use CLI args or relative paths
- No `print()` for debugging — use proper logging or remove before commit
- No wildcard imports (`from x import *`)
- No GPU-dependent code without CPU fallback

## Key Design Decisions

- Input features: 62 EEG channels × 5 freq bands = 310 features
- SNN architecture: 310 → 256 → 128 → 3 (LIF neurons, learnable beta)
- Spike encoding: rate coding via spikegen.rate(), T=25 timesteps
- Readout: spike count over timesteps → argmax for prediction
- Synthetic data must produce ~33% accuracy (random chance, 3 classes) — this is correct
- SEED .mat files may be v7.3 HDF5 — try scipy.io.loadmat first, fall back to h5py

## Author

Tejas J Bharadwaj, CMU SCS Class of 2030. README must include name, CMU affiliation, and social links.
