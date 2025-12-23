import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
data_path = Path(__file__).parent / "similarities.jsonl"
similarities = []

with open(data_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        similarities.append(obj['similarities'])

similarities = np.array(similarities)
print(f"Data shape: {similarities.shape}")  # (80, 10)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Heatmap
ax1 = axes[0, 0]
sns.heatmap(similarities[:20], cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Similarity Score'})
ax1.set_title('Similarity Heatmap (First 20 samples)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Comparison Index')
ax1.set_ylabel('Sample Index')

# 2. Distribution histogram
ax2 = axes[0, 1]
flattened = similarities.flatten()
ax2.hist(flattened, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(np.mean(flattened), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(flattened):.3f}')
ax2.axvline(np.median(flattened), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(flattened):.3f}')
ax2.set_title('Distribution of All Similarity Scores', fontsize=12, fontweight='bold')
ax2.set_xlabel('Similarity Score')
ax2.set_ylabel('Frequency')
ax2.legend()

# 3. Box plot
ax3 = axes[1, 0]
ax3.boxplot(similarities, vert=True)
ax3.set_title('Box Plot of Similarities per Comparison', fontsize=12, fontweight='bold')
ax3.set_xlabel('Comparison Index (1-10)')
ax3.set_ylabel('Similarity Score')
ax3.grid(True, alpha=0.3)

# 4. Line plot - mean similarity trend
ax4 = axes[1, 1]
mean_per_idx = np.mean(similarities, axis=0)
std_per_idx = np.std(similarities, axis=0)
x = np.arange(len(mean_per_idx))
ax4.plot(x, mean_per_idx, marker='o', linewidth=2, markersize=8, color='steelblue', label='Mean')
ax4.fill_between(x, mean_per_idx - std_per_idx, mean_per_idx + std_per_idx, alpha=0.3, color='steelblue', label='±1 Std')
ax4.set_title('Average Similarity Trend', fontsize=12, fontweight='bold')
ax4.set_xlabel('Comparison Index')
ax4.set_ylabel('Average Similarity Score')
ax4.set_xticks(x)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "similarities_visualization.png", dpi=300, bbox_inches='tight')
print("✓ Saved: similarities_visualization.png")
plt.show()

# Print statistics
print("\n" + "="*50)
print("STATISTICS")
print("="*50)
print(f"Total samples: {similarities.shape[0]}")
print(f"Comparisons per sample: {similarities.shape[1]}")
print(f"Overall mean: {np.mean(flattened):.4f}")
print(f"Overall median: {np.median(flattened):.4f}")
print(f"Overall std: {np.std(flattened):.4f}")
print(f"Min: {np.min(flattened):.4f}")
print(f"Max: {np.max(flattened):.4f}")
