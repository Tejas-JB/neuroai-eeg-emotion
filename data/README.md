# SEED Dataset — Download Instructions

This project trains on the **SEED** (SJTU Emotion EEG Dataset) produced by the
[BCMI Lab at Shanghai Jiao Tong University](https://bcmi.sjtu.edu.cn/home/seed/).
The dataset is free for academic use but requires a registration request.

> **The pipeline runs end-to-end on synthetic data without SEED.** Use
> `--use-synthetic` on any command in the top-level README to try it immediately.
> SEED is only required to reproduce real-data results.

## 1. Request access

1. Visit [https://bcmi.sjtu.edu.cn/home/seed/](https://bcmi.sjtu.edu.cn/home/seed/).
2. Fill out the SEED access form using an **academic email address**.
3. Wait for the download credentials — the lab usually responds within a few days.

## 2. Download

Only the `ExtractedFeatures_1s/` folder is required — it contains the
per-session Differential Entropy (DE) feature `.mat` files that this pipeline
consumes, computed on 1-second windows. (The 4-second `ExtractedFeatures/`
variant also works; pass `--data-path data/ExtractedFeatures/`.) You do
**not** need the raw `Preprocessed_EEG/` signals.

`label.mat` ships inside `ExtractedFeatures_1s/`; the loader picks it up
automatically.

## 3. Place the files

The expected directory layout under this project root:

```
data/
├── README.md                    # this file
└── ExtractedFeatures_1s/
    ├── label.mat                # 15 labels in {-1, 0, +1}
    ├── readme.txt               # SEED citation
    ├── 1_20131027.mat
    ├── 2_20140404.mat
    ├── ...                      # 45 session files total
    └── 15_20131105.mat
```

The loader (`src/data_loader.py`) handles both the standard `scipy.io.loadmat`
format and MATLAB v7.3 (HDF5) via an automatic `h5py` fallback. It inspects
keys and reshapes each trial's DE array to `(N_windows, 310)` at runtime —
you do not need to pre-process anything.

## 4. Run on real data

From the project root:

```bash
python src/train.py --model snn      --epochs 50 --data-path data/ExtractedFeatures_1s/
python src/evaluate.py --model snn   --checkpoint results/snn_checkpoint.pt \
                                     --data-path data/ExtractedFeatures_1s/
```

See the top-level README's **Training on Real Data** section for the full
workflow (baseline training, visualization regeneration, hyperparameter tips).

## 5. Licensing and redistribution

SEED is released for **academic, non-commercial research only**. Redistribution
is **not** permitted. `.gitignore` excludes any `.mat` files under `data/` so
you will never accidentally commit them.

## 6. Citation

Please cite the original SEED paper when using the dataset:

```bibtex
@article{zheng2015investigating,
  title   = {Investigating Critical Frequency Bands and Channels for {EEG}-based Emotion Recognition with Deep Neural Networks},
  author  = {Zheng, Wei-Long and Lu, Bao-Liang},
  journal = {IEEE Transactions on Autonomous Mental Development},
  volume  = {7},
  number  = {3},
  pages   = {162--175},
  year    = {2015},
  publisher = {IEEE}
}
```
