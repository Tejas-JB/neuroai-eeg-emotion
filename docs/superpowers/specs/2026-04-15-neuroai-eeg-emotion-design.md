# Design Spec — neuroai-eeg-emotion

**Date:** 2026-04-15
**Author:** Tejas J Bharadwaj (CMU SCS '30)
**Status:** Approved — proceeding to phased implementation

---

## 1. Purpose

Build a reproducible, open-source pipeline that classifies emotions (positive / neutral / negative) from SEED EEG differential-entropy features using a Spiking Neural Network (snnTorch, LIF neurons), compared against a parameter-matched MLP baseline. Full spec lives in `docs/neuroai-eeg-emotion-PRD.md` and `docs/neuroai-eeg-emotion-ACTION-PLAN.md`; this document records the orchestration design layered on top.

## 2. Repo & Git Architecture

- **Repo:** `github.com/Tejas-JB/neuroai-eeg-emotion` (public, MIT)
- **Location:** in-place at `/Users/Tejas/Documents/SEED/` (repo root)
- **Trunk-based:** `main` is always green; each phase lands via short-lived `phase/NN-<slug>` branch + squash-merged PR + semver-ish tag `v0.N-phaseN`.
- **Tags:** `v0.1-phase1` … `v0.8-phase8`; final integration tagged `v1.0.0`.
- **Branch protection:** deferred (optional); PR-driven flow used voluntarily for review discipline.

## 3. Phase Map (8 phases + Phase 0 bootstrap)

| Phase | Branch | Deliverable | Tag |
|---|---|---|---|
| 0 | `main` | repo init, GitHub remote, initial docs commit | — |
| 1 | `phase/01-scaffold` | dirs, deps installed, LICENSE, .gitignore | v0.1-phase1 |
| 2 | `phase/02-data-loader` | synthetic + SEED `.mat` loader + unified `load_data` | v0.2-phase2 |
| 3 | `phase/03-spike-encoding` | rate + delta encoders, normalization, demo viz | v0.3-phase3 |
| 4 | `phase/04-models` | SNN (LIF) + baseline MLP classes with forward-pass tests | v0.4-phase4 |
| 5 | `phase/05-training` | `train.py` for both models, checkpoint + training_curves.png | v0.5-phase5 |
| 6 | `phase/06-eval-viz` | evaluate.py + visualize.py (4 required viz types) | v0.6-phase6 |
| 7 | `phase/07-docs` | full README + data/README.md + prohibited-term gate | v0.7-phase7 |
| 8 | `phase/08-integration` | demo.ipynb + full integration test + flake8 clean | v1.0.0 |

Each phase maps 1:1 to ACTION-PLAN checkpoints; PR merges only after checkpoint artifacts verified.

## 4. Agent Dispatch Strategy

- **Main thread:** scaffolding, code authoring, git ops, PR creation.
- **`feature-dev:code-reviewer`:** pre-PR checkpoint audit against PRD spec (phases 3–8).
- **`Explore` (thorough):** snnTorch API verification, SEED .mat key inspection (phases 2, 3, 4).
- **`context7` MCP:** fetch current snnTorch docs to avoid training-data drift.
- **Worktrees:** not used for this build — phases are mostly sequential; added complexity not justified.

## 5. Constraints (from PRD)

- **Hard prohibitions** (grep-gated in Phase 7): Presonance, TRIBE v2, fMRI, neuromarketing.
- Dark matplotlib theme (PRD §8) applied via `setup_dark_theme()` called before every plot.
- Synthetic-data fallback must keep whole pipeline runnable with zero external deps.
- All CLI scripts usable with zero flags (sensible defaults).
- MIT license, CMU/Tejas attribution in README.

## 6. Key Technical Contracts

- Features: 62 channels × 5 bands = **310**.
- SNN shape: 310 → 256 → 128 → 3; LIF with learnable β, threshold 1.0, fast_sigmoid surrogate.
- Spike encoder output: `(T, batch, 310)` binary, T=25 default.
- Readout: sum spikes over T → argmax.
- MLP matches layer dims and approximate param count for fair comparison.
- Labels: {-1, 0, +1} in SEED → remapped to {0, 1, 2} internally.

## 7. Success Gate

`v1.0.0` ships when:
- All 8 phase tags exist on `main`.
- `python src/train.py --model snn --epochs 5 --use-synthetic` runs cold-clone in under 5 min.
- All 8 PNGs exist in `results/` at non-zero size.
- `flake8 src/ --max-line-length=120 --ignore=E501,W503` passes.
- `grep -r "presonance\|TRIBE\|fMRI\|neuromarketing" --include="*.py" --include="*.md" .` returns empty.
- README renders with Tejas/CMU attribution and all viz embedded.

## 8. Out of Scope (deferred)

Real-data training (GCP, post-build), neuromorphic HW deployment, real-time inference, raw-signal processing, web/mobile. All per PRD §2.3 non-goals.
