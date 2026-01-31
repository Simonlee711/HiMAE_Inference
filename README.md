![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ulzee/raptor-private/pulls)
[![license](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/Simonlee711/HiMAE/blob/main/LICENSE)
[![release](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/Simonlee711/HiMAE/releases/tag/v0.1.0)

# <center> 🔎 HiMAE: Hierarchical Masked Auto Encoders 🔍 </center>

![raptor-banner](https://github.com/Simonlee711/HiMAE/blob/main/img/heatmap-himae.png?raw=true)

**[ICLR 2026]** HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series

**Authors:** [Simon A. Lee](https://simon-a-lee.github.io), Cyrus Tanade, Hao Zhou, Juhyeon Lee, Megha Thukral, Minji Han, Rachel Choi, Md Sazzad Hissain Khan, Baiying Lu, Migyeong Gwak, Mehrab Bin Morshed, Viswam Nathan, Md Mahbubur Rahman, Li Zhu, Subramaniam Venkatraman, Sharanya Arcot Desai

---

# Brief Description

Self-supervised masked autoencoding for physiological waveforms with HiMAE for PVC detection. This repository contains a PyTorch/Lightning implementation of a hierarchical 1‑D convolutional MAE (“HiMAE”), a minimal pretraining script, and a reproducible linear‑probe pipeline on 10‑second PPG segments.


## What’s in the repo

The root directory includes a pretrain checkpoint, a reference linear probe, a small metadata CSV for the synthetic PVC task, and a demonstration notebook.

```
HiMAE_PVC_Detection.ipynb        ← end‑to‑end wiring for PVC linear probe
himae_synth.ckpt                 ← Lightning checkpoint for HiMAE backbone
pvc_linear_probe.pt              ← state_dict for reference linear probe
pvc_10s_synth_metadata.csv       ← example metadata (fs=25 Hz, 10 s windows)
pvc_predictions.csv              ← example inference outputs (p_pvc per segment)
pretrain/
himae.py                       ← minimal Lightning trainer for masked AE
pvc/
utils/                         ← logger and model registry
helper_logger.py
helper_models.py
model_arch/himae.py          ← 1‑D CNN HiMAE backbone (encoder/decoder)
downstream_eval/
binary_linear_prob.py        ← script for linear probe training/eval
helpers.py                   ← analysis utilities
LICENSE
README.md

````

---

## Installation

Use Python 3.10+ with CUDA‑enabled PyTorch if available. A compact setup is below; choose the CUDA index URL that matches your system.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.*" "torchvision==0.*" "torchaudio==2.*" --index-url https://download.pytorch.org/whl/cu121
pip install lightning pytorch-lightning torchmetrics h5py s3fs boto3 pandas numpy tabulate matplotlib scikit-learn pyyaml wandb
````

W&B logging is enabled by default in pretraining; set `WANDB_DISABLED=true` if you prefer to run offline.

---

## Data format

Pretraining expects a CSV that indexes samples stored in HDF5 shards. Each row references a shard path and a sample key with a `normalized_waveform` dataset:

```
local_path,global_idx
/path/to/shard_A.h5,000123
/path/to/shard_B.h5,000987
...
```

Each `h5py.File(local_path)[global_idx]['normalized_waveform'][:]` should yield a 1‑D float array of length (L=f_s\times T).

Downstream PVC uses an HDF5 with contiguous datasets for signals and labels, for example:

* `/ppg` with shape `[N, L]` or `/ecg` with `[N, L]`
* `/labels` with shape `[N]` (binary), and optionally `/patient_ids` with `[N]`

The included `pvc_10s_synth_metadata.csv` advertises segments sampled at 25 Hz with 10‑second windows ((L=250)) and a binary `pvc` label. The demo notebook shows how to feed either such an HDF5 or synthetic tensors into the probe.

---

## PVC linear probe

The PVC probe freezes the encoder and fits a single logistic layer on top of mean‑pooled bottleneck features. The simplest path is the Jupyter notebook:

1. Open `HiMAE_PVC_Detection.ipynb` and set `H5_PATH`, `META_PATH` (optional), `SIGNAL_KEY` (`ppg` or `ecg`), and the `CFG` block. The included configuration for the synthetic data uses (f_s=25) Hz and (T=10) s.
2. Point the backbone to `himae_synth.ckpt` and the probe to `pvc_linear_probe.pt` (or train a fresh probe in a few epochs).
3. Run the training and evaluation cells. The notebook will optionally write `pvc_predictions.csv` with patient IDs, labels, and predicted probabilities.

If you prefer a pure‑script flow, `pvc/downstream_eval/binary_linear_prob.py` contains the same logic. The script includes S3 helpers; for local files, wire `_read_one_h5_from_local` to your path and construct the `cfg` dict as in the notebook.

---

## Reference results

The included `pvc_predictions.csv` contains 11,172 synthetic segments with a PVC prevalence of 4.61% (515 positives). Using the provided backbone and a simple linear probe, the aggregate metrics on that split are:

* ROC‑AUC ≈ **0.766**

These values reflect a highly imbalanced binary task and a deliberately minimal probe. They serve as a sanity‑check rather than a saturated benchmark.

---

## Reproducing and extending

The repository is intentionally modular. To adapt to new tasks, point the metadata to your HDF5 shards, adjust `sampling_freq` and `seg_len` accordingly, and keep the masked reconstruction loss unchanged. The bottleneck dimensionality is 256 by default; if you change the encoder channels, update the probe input size to match. For longer segments, consider proportionally increasing the depth to keep the bottleneck time resolution reasonable after stride‑2 downsamples.

---

## BibTeX

If this code is useful in your work, please cite the repository:

`ICLR`
```
@inproceedings{lee2025himae,
  title={HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series},
  author={Lee, Simon A and Tanade, Cyrus and Zhou, Hao and Lee, Juhyeon and Thukral, Megha and Khan, Md Sazzad Hissain and Lu, Baiying and Gwak, Migyeong and others},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=iPAy5VpGQa}
}
```

`arxiv`
```
@article{lee2025himae,
  title={HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series},
  author={Lee, Simon A and Tanade, Cyrus and Zhou, Hao and Lee, Juhyeon and Thukral, Megha and Han, Minji and Choi, Rachel and Khan, Md Sazzad Hissain and Lu, Baiying and Gwak, Migyeong and others},
  journal={arXiv preprint arXiv:2510.25785},
  year={2025}
}
```

## Acknowledgements

We thank Minji Han and Rachel Choi for their expertise in UX/UI design and for crafting the specialized visualizations not supported by standard Python libraries; their design contributions were essential to this work. We also thank Praveen Raja, Matthew Wiggins, and Mike Freedman for their invaluable feedback and insightful discussions throughout the project.
