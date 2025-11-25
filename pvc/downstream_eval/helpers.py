# This cell defines a utility to summarize patient-level binary labels,
# prints a Markdown table using `tabulate`, and saves multiple artifacts locally.
# Replace the demo data at the bottom with your real arrays/lists.

import os
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# For nice DataFrame display in the UI if helpful
# from caas_jupyter_tools import display_dataframe_to_user

def summarize_dataset(patient_ids, labels, args):
    """
    Summarize binary labels (0/1) grouped by patient_id.
    
    Parameters
    ----------
    patient_ids : array-like (len N)
        Patient identifier per row/segment (hashable: str/int).
    labels : array-like (len N)
        Binary labels (0 or 1) per row/segment.
    out_dir : str
        Directory to save outputs (created if missing).
    min_count_for_rate_rank : int
        When ranking patients by positive rate, require at least this many segments.
    
    Returns
    -------
    artifacts : dict
        Paths to saved files and key summary statistics.
    """
    min_count_for_rate_rank = 5
    out_dir = f"downstream_eval/{args.app}/"
    os.makedirs(out_dir, exist_ok=True)
    # print(patient_ids)
    
    # Convert to numpy arrays
    pids = np.asarray(patient_ids)
    y = np.asarray(labels).astype(int)
    assert pids.shape[0] == y.shape[0], "patient_ids and labels must have same length"
    assert set(np.unique(y)).issubset({0,1}), "labels must be binary (0/1)"
    
    N = len(y)
    unique_patients, counts = np.unique(pids, return_counts=True)
    # print(unique_patients)
    P = len(unique_patients)
    pos = int(y.sum())
    neg = int((1 - y).sum())
    pos_rate = pos / N if N else float("nan")
    
    # Majority-class baseline (always predict the most common class)
    majority_class = int(pos >= neg)
    baseline_acc = max(pos, neg) / N if N else float("nan")
    # Entropy of labels (bits) & Gini impurity (useful to understand class uncertainty)
    eps = 1e-12
    p1 = pos_rate
    p0 = 1 - pos_rate
    entropy = -(p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))
    gini = 1 - (p1**2 + p0**2)
    
    # Per-patient aggregates
    # counts per patient
    by_patient = defaultdict(list)
    for pid, lab in zip(pids, y):
        by_patient[pid].append(lab)
    
    rows = []
    for pid, labs in by_patient.items():
        arr = np.asarray(labs, dtype=int)
        n = arr.size
        s = int(arr.sum())
        z = int(n - s)
        rate = s / n if n else float("nan")
        rows.append([pid, n, s, z, rate])
    
    per_patient_df = pd.DataFrame(rows, columns=["patient_id", "n_segments", "n_pos", "n_neg", "pos_rate"]).sort_values("patient_id")
    
    # Patients categories
    n_any_pos = int((per_patient_df["n_pos"] > 0).sum())
    n_all_neg = int((per_patient_df["n_pos"] == 0).sum())
    n_all_pos = int((per_patient_df["n_neg"] == 0).sum())
    n_mixed   = int(((per_patient_df["n_pos"] > 0) & (per_patient_df["n_neg"] > 0)).sum())
    
    # Distribution summaries
    segs_per_patient = per_patient_df["n_segments"].to_numpy()
    rate_per_patient = per_patient_df["pos_rate"].to_numpy()
    
    def q(x, p): 
        return float(np.nanquantile(x, p)) if x.size else float("nan")
    
    segs_stats = {
        "mean": float(np.nanmean(segs_per_patient)) if segs_per_patient.size else float("nan"),
        "std": float(np.nanstd(segs_per_patient)) if segs_per_patient.size else float("nan"),
        "min": float(np.nanmin(segs_per_patient)) if segs_per_patient.size else float("nan"),
        "p25": q(segs_per_patient, 0.25),
        "p50": q(segs_per_patient, 0.50),
        "p75": q(segs_per_patient, 0.75),
        "max": float(np.nanmax(segs_per_patient)) if segs_per_patient.size else float("nan"),
    }
    rate_stats = {
        "mean": float(np.nanmean(rate_per_patient)) if rate_per_patient.size else float("nan"),
        "std": float(np.nanstd(rate_per_patient)) if rate_per_patient.size else float("nan"),
        "min": float(np.nanmin(rate_per_patient)) if rate_per_patient.size else float("nan"),
        "p10": q(rate_per_patient, 0.10),
        "p25": q(rate_per_patient, 0.25),
        "p50": q(rate_per_patient, 0.50),
        "p75": q(rate_per_patient, 0.75),
        "p90": q(rate_per_patient, 0.90),
        "max": float(np.nanmax(rate_per_patient)) if rate_per_patient.size else float("nan"),
    }
    
    # Rank patients
    # Top/bottom by number of positives
    top_pos = per_patient_df.sort_values(["n_pos", "n_segments"], ascending=[False, False]).head(10)
    bottom_pos = per_patient_df.sort_values(["n_pos", "n_segments"], ascending=[True, True]).head(10)
    # Top by pos_rate with min_count_for_rate_rank
    eligible = per_patient_df[per_patient_df["n_segments"] >= min_count_for_rate_rank]
    top_by_rate = eligible.sort_values(["pos_rate", "n_segments"], ascending=[False, False]).head(10)
    bottom_by_rate = eligible.sort_values(["pos_rate", "n_segments"], ascending=[True, True]).head(10)
    
    # Save per-patient table
    per_patient_csv = os.path.join(out_dir, "per_patient_stats.csv")
    per_patient_df.to_csv(per_patient_csv, index=False)
    
    # Make a histogram of per-patient positive rate
    if len(eligible) > 0:
        plt.figure()
        plt.hist(eligible["pos_rate"], bins=20)
        plt.xlabel("Per-patient positive rate")
        plt.ylabel("Count of patients")
        plt.title(f"Distribution of per-patient positive rate (n≥{min_count_for_rate_rank})")
        rate_hist_png = os.path.join(out_dir, "pos_rate_hist.png")
        plt.tight_layout()
        plt.savefig(rate_hist_png, dpi=150)
        plt.close()
    else:
        rate_hist_png = None
    
    # Build summary table (Markdown via tabulate)
    summary_rows = [
        ["Total rows (segments)", N],
        ["Unique patients", P],
        ["Total positives", pos],
        ["Total negatives", neg],
        ["Overall positive rate", f"{pos_rate*100:.2f}%"],
        ["Majority class", majority_class],
        ["Majority-class baseline accuracy", f"{baseline_acc*100:.2f}%"],
        ["Label entropy (bits)", f"{entropy:.4f}"],
        ["Label Gini impurity", f"{gini:.4f}"],
        ["Patients with ≥1 positive", n_any_pos],
        ["Patients with only 0's", n_all_neg],
        ["Patients with only 1's", n_all_pos],
        ["Patients with mixed labels", n_mixed],
        ["Avg segments per patient (mean±std)", f"{segs_stats['mean']:.2f} ± {segs_stats['std']:.2f}"],
        ["Segments per patient (min/25/50/75/max)", f"{segs_stats['min']:.0f} / {segs_stats['p25']:.0f} / {segs_stats['p50']:.0f} / {segs_stats['p75']:.0f} / {segs_stats['max']:.0f}"],
        ["Per-patient positive rate mean±std", f"{rate_stats['mean']:.3f} ± {rate_stats['std']:.3f}"],
        ["Per-patient positive rate (min/10/25/50/75/90/max)", f"{rate_stats['min']:.3f} / {rate_stats['p10']:.3f} / {rate_stats['p25']:.3f} / {rate_stats['p50']:.3f} / {rate_stats['p75']:.3f} / {rate_stats['p90']:.3f} / {rate_stats['max']:.3f}"],
    ]
    summary_md = tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="github")
    print(summary_md)
    
    # Save the markdown summary and the top/bottom tables
    summary_md_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_md_path, "w") as f:
        f.write("# Patient Label Summary\n\n")
        f.write(summary_md + "\n\n")
        f.write("## Top 10 patients by number of positives\n\n")
        f.write(tabulate(top_pos.values.tolist(), headers=top_pos.columns.tolist(), tablefmt="github"))
        f.write("\n\n## Bottom 10 patients by number of positives\n\n")
        f.write(tabulate(bottom_pos.values.tolist(), headers=bottom_pos.columns.tolist(), tablefmt="github"))
        f.write(f"\n\n## Top 10 by positive rate (n ≥ {min_count_for_rate_rank})\n\n")
        f.write(tabulate(top_by_rate.values.tolist(), headers=top_by_rate.columns.tolist(), tablefmt="github"))
        f.write(f"\n\n## Bottom 10 by positive rate (n ≥ {min_count_for_rate_rank})\n\n")
        f.write(tabulate(bottom_by_rate.values.tolist(), headers=bottom_by_rate.columns.tolist(), tablefmt="github"))
        if rate_hist_png:
            f.write(f"\n\n![Per-patient positive rate histogram]({os.path.basename(rate_hist_png)})\n")
    
    # Also save JSON with key stats
    summary_json_path = os.path.join(out_dir, "summary.json")
    key_stats = {
        "N_rows": N,
        "unique_patients": P,
        "total_pos": pos,
        "total_neg": neg,
        "overall_pos_rate": pos_rate,
        "baseline_majority_class": majority_class,
        "baseline_accuracy": baseline_acc,
        "entropy_bits": float(entropy),
        "gini_impurity": float(gini),
        "patients_any_pos": n_any_pos,
        "patients_all_neg": n_all_neg,
        "patients_all_pos": n_all_pos,
        "patients_mixed": n_mixed,
        "segs_per_patient_stats": segs_stats,
        "pos_rate_per_patient_stats": rate_stats,
        "min_count_for_rate_rank": min_count_for_rate_rank,
        "artifacts": {
            "per_patient_csv": per_patient_csv,
            "summary_md": summary_md_path,
            "rate_hist_png": rate_hist_png,
        }
    }
    with open(summary_json_path, "w") as jf:
        json.dump(key_stats, jf, indent=2)
    



