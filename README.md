# HiMAE ICLR Synthetic PVC Example

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

The included `pvc_predictions.csv` contains 11,172 segments with a PVC prevalence of 4.61% (515 positives). Using the provided backbone and a simple linear probe, the aggregate metrics on that split are:

* ROC‑AUC ≈ **0.766**
* Average Precision ≈ **0.116**

These values reflect a highly imbalanced binary task and a deliberately minimal probe. They serve as a sanity‑check rather than a saturated benchmark.

---

## Reproducing and extending

The repository is intentionally modular. To adapt to new tasks, point the metadata to your HDF5 shards, adjust `sampling_freq` and `seg_len` accordingly, and keep the masked reconstruction loss unchanged. The bottleneck dimensionality is 256 by default; if you change the encoder channels, update the probe input size to match. For longer segments, consider proportionally increasing the depth to keep the bottleneck time resolution reasonable after stride‑2 downsamples.

---

## BibTeX

If this code is useful in your work, please cite the repository. Replace the placeholders as needed.

> included upon acceptance

## Acknowledgements

Built with PyTorch, Lightning, and TorchMetrics. Optional logging uses Weights & Biases. Many implementation choices were guided by prior work on masked autoencoders adapted to 1‑D physiological signals.
