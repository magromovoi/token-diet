import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results")
REPORT = Path("images")
REPORT.mkdir(exist_ok=True)

CONDITIONS = ["baseline", "output_compressed", "input_compressed", "both"]
LABELS = ["Baseline", "Out compr", "In compr", "Both"]

dirs = sorted(OUT.glob("product_*"))
data = {}
for cond in CONDITIONS:
    rows = []
    for d in dirs:
        p = d / f"{cond}.json"
        if p.exists(): rows.append(json.load(open(p)))
    data[cond] = rows

mean_inp = [np.mean([r["input_tokens"] for r in data[c]]) for c in CONDITIONS]
mean_out = [np.mean([r["output_tokens"] for r in data[c]]) for c in CONDITIONS]
costs = [(np.sum([r["input_tokens"] for r in data[c]]) * 3.0 +
          np.sum([r["output_tokens"] for r in data[c]]) * 15.0) / 1e6 for c in CONDITIONS]
bert = [np.mean([r["bertscore_f1"] for r in data[c] if r.get("bertscore_f1")]) for c in CONDITIONS]
x = np.arange(4)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].bar(x, mean_inp)
axes[0].set_xticks(x); axes[0].set_xticklabels(LABELS, fontsize=8)
axes[0].set_ylabel("Mean input tokens"); axes[0].set_title("Input tokens per sample")

axes[1].bar(x, mean_out)
axes[1].set_xticks(x); axes[1].set_xticklabels(LABELS, fontsize=8)
axes[1].set_ylabel("Mean output tokens"); axes[1].set_title("Output tokens per sample")

axes[2].bar(x, costs)
axes[2].set_xticks(x); axes[2].set_xticklabels(LABELS, fontsize=8)
axes[2].set_ylabel("Cost ($)"); axes[2].set_title("Total cost (150 samples)")
plt.tight_layout()
plt.savefig(REPORT / "token_cost_bars.png", dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(5, 3.5))
for i, cond in enumerate(CONDITIONS):
    ax.scatter(costs[i], bert[i], s=120, zorder=5, label=LABELS[i])
ax.set_xlabel("Total cost ($)"); ax.set_ylabel("Mean BERTScore F1")
ax.set_title("Cost-quality tradeoff")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT / "cost_vs_bertscore.png", dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(5, 3.5))
bp_data = [[r["bertscore_f1"] for r in data[c] if r.get("bertscore_f1")] for c in CONDITIONS]
bp = ax.boxplot(bp_data, tick_labels=LABELS, patch_artist=True,
                medianprops=dict(color="black"))
for patch in bp["boxes"]:
    patch.set_alpha(0.6)
ax.set_ylabel("BERTScore F1"); ax.set_title("Quality distribution across conditions")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(REPORT / "bertscore_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()


out_baseline = [r["output_tokens"] for r in data["baseline"]]
out_caveman = [r["output_tokens"] for r in data["output_compressed"]]
savings_pct = [(b - c) / b * 100 for b, c in zip(out_baseline, out_caveman)]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(out_baseline, savings_pct, s=15, alpha=0.6)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Baseline output tokens")
ax.set_ylabel("Output token savings (%)")
ax.set_title("Caveman savings vs. baseline verbosity")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT / "caveman_savings_scatter.png", dpi=150, bbox_inches="tight")
plt.close()


ratios = []
for d in dirs:
    ic = json.load(open(d / "input_compressed.json"))
    b = json.load(open(d / "baseline.json"))
    ratios.append(ic["input_tokens"] / b["input_tokens"])

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(ratios, bins=20, alpha=0.75, edgecolor="white")
ax.axvline(np.mean(ratios), color="black", linestyle="--", label=f"Mean: {np.mean(ratios):.2f}")
ax.set_xlabel("Compression ratio (compressed / original)")
ax.set_ylabel("Count"); ax.set_title("Input compression ratio distribution")
ax.legend()
plt.tight_layout()
plt.savefig(REPORT / "compression_ratio_hist.png", dpi=150, bbox_inches="tight")
plt.close()
