#!/usr/bin/env python3
"""
Plot Kendrick Mass Defect (KMD) Results for PFAS Identification

Generates visualization plots showing:
1. KMD vs Kendrick Mass (classic KMD plot)
2. KMD distribution for PFAS vs non-PFAS
3. Score component contributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(filepath):
    """Load predictions from TSV file."""
    print(f"Loading results from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} predictions")
    return df

def plot_kmd_vs_kendrick_mass(df, output_path):
    """
    Plot KMD vs Kendrick Mass (classic KMD plot).
    
    PFAS compounds with the same functional group but different CF2 chain lengths
    will align horizontally (same KMD, different Kendrick mass).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Separate actual PFAS vs non-PFAS
    pfas = df[df['is_PFAS'] == True]
    non_pfas = df[df['is_PFAS'] == False]
    
    # Plot non-PFAS first (background)
    ax.scatter(non_pfas['kendrick_mass'], non_pfas['kmd'], 
              alpha=0.3, s=10, c='gray', label=f'Non-PFAS (n={len(non_pfas)})')
    
    # Plot actual PFAS
    ax.scatter(pfas['kendrick_mass'], pfas['kmd'], 
              alpha=0.6, s=20, c='red', label=f'Actual PFAS (n={len(pfas)})')
    
    # Add KMD threshold lines
    ax.axhline(y=0.15, color='blue', linestyle='--', linewidth=1.5, 
               label='KMD threshold (±0.15)')
    ax.axhline(y=-0.15, color='blue', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Kendrick Mass (CF₂ base)', fontsize=12)
    ax.set_ylabel('Kendrick Mass Defect', fontsize=12)
    ax.set_title('Kendrick Mass Defect Plot: PFAS vs Non-PFAS\n' + 
                 'Homologous series (CF₂ repeats) align horizontally', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    ax.text(0.02, 0.98, 
            f'PFAS typically cluster near KMD=0\nIndicating CF₂ homologous series',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved KMD plot to {output_path}")
    plt.close()

def plot_kmd_distribution(df, output_path):
    """Plot KMD distribution comparing PFAS vs non-PFAS."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Separate by actual PFAS status
    pfas = df[df['is_PFAS'] == True]
    non_pfas = df[df['is_PFAS'] == False]
    
    # Plot 1: Histogram comparison
    ax1 = axes[0]
    bins = np.linspace(-0.5, 0.5, 100)
    ax1.hist(non_pfas['kmd'], bins=bins, alpha=0.6, label=f'Non-PFAS (n={len(non_pfas)})', 
             color='gray', density=True)
    ax1.hist(pfas['kmd'], bins=bins, alpha=0.7, label=f'Actual PFAS (n={len(pfas)})', 
             color='red', density=True)
    
    # Add threshold lines
    ax1.axvline(x=0.15, color='blue', linestyle='--', linewidth=2, label='KMD threshold (±0.15)')
    ax1.axvline(x=-0.15, color='blue', linestyle='--', linewidth=2)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax1.set_xlabel('Kendrick Mass Defect', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('KMD Distribution: PFAS vs Non-PFAS', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate statistics
    pfas_mean_kmd = pfas['kmd'].mean()
    non_pfas_mean_kmd = non_pfas['kmd'].mean()
    pfas_within_threshold = (np.abs(pfas['kmd']) <= 0.15).sum() / len(pfas) * 100
    non_pfas_within_threshold = (np.abs(non_pfas['kmd']) <= 0.15).sum() / len(non_pfas) * 100
    
    stats_text = f'Mean KMD:\n  PFAS: {pfas_mean_kmd:.4f}\n  Non-PFAS: {non_pfas_mean_kmd:.4f}\n\n'
    stats_text += f'Within threshold:\n  PFAS: {pfas_within_threshold:.1f}%\n  Non-PFAS: {non_pfas_within_threshold:.1f}%'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Box plot comparison
    ax2 = axes[1]
    data_to_plot = [non_pfas['kmd'].abs(), pfas['kmd'].abs()]
    bp = ax2.boxplot(data_to_plot, labels=['Non-PFAS', 'Actual PFAS'], 
                     patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['gray', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.axhline(y=0.15, color='blue', linestyle='--', linewidth=2, label='KMD threshold (0.15)')
    ax2.set_ylabel('|Kendrick Mass Defect|', fontsize=12)
    ax2.set_title('Absolute KMD Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved KMD distribution plot to {output_path}")
    plt.close()

def plot_score_contributions(df, output_path):
    """Plot how different scoring components contribute to predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Separate by prediction
    predicted_pfas = df[df['predicted_pfas'] == True]
    predicted_non_pfas = df[df['predicted_pfas'] == False]
    
    # Plot 1: Score component stacked bar
    ax1 = axes[0, 0]
    components = ['cf2_score', 'fragment_score', 'kmd_score']
    means_pfas = [predicted_pfas[c].mean() for c in components]
    means_non_pfas = [predicted_non_pfas[c].mean() for c in components]
    
    x = np.arange(len(components))
    width = 0.35
    
    ax1.bar(x - width/2, means_pfas, width, label='Predicted PFAS', color='red', alpha=0.7)
    ax1.bar(x + width/2, means_non_pfas, width, label='Predicted Non-PFAS', color='gray', alpha=0.7)
    
    ax1.set_ylabel('Average Score', fontsize=11)
    ax1.set_title('Average Score Components', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['CF₂ Score', 'Fragment Score', 'KMD Score'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Scatter of CF2 score vs KMD score
    ax2 = axes[0, 1]
    # Sample for clarity (too many points)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    pfas_sample = df_sample[df_sample['is_PFAS'] == True]
    non_pfas_sample = df_sample[df_sample['is_PFAS'] == False]
    
    ax2.scatter(non_pfas_sample['cf2_score'], non_pfas_sample['kmd_score'], 
               alpha=0.3, s=20, c='gray', label='Non-PFAS')
    ax2.scatter(pfas_sample['cf2_score'], pfas_sample['kmd_score'], 
               alpha=0.6, s=30, c='red', label='Actual PFAS')
    
    ax2.set_xlabel('CF₂ Score', fontsize=11)
    ax2.set_ylabel('KMD Score', fontsize=11)
    ax2.set_title(f'CF₂ vs KMD Score (n={sample_size} sample)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection method pie chart for predicted PFAS
    ax3 = axes[1, 0]
    
    kmd_only = len(predicted_pfas[(predicted_pfas['kmd_score'] > 0) & 
                                  (predicted_pfas['cf2_score'] == 0) & 
                                  (predicted_pfas['fragment_score'] == 0)])
    cf2_only = len(predicted_pfas[(predicted_pfas['cf2_score'] > 0) & 
                                  (predicted_pfas['fragment_score'] == 0) & 
                                  (predicted_pfas['kmd_score'] == 0)])
    frag_only = len(predicted_pfas[(predicted_pfas['fragment_score'] > 0) & 
                                   (predicted_pfas['cf2_score'] == 0) & 
                                   (predicted_pfas['kmd_score'] == 0)])
    cf2_kmd = len(predicted_pfas[(predicted_pfas['cf2_score'] > 0) & 
                                 (predicted_pfas['kmd_score'] > 0) & 
                                 (predicted_pfas['fragment_score'] == 0)])
    cf2_frag = len(predicted_pfas[(predicted_pfas['cf2_score'] > 0) & 
                                  (predicted_pfas['fragment_score'] > 0) & 
                                  (predicted_pfas['kmd_score'] == 0)])
    frag_kmd = len(predicted_pfas[(predicted_pfas['fragment_score'] > 0) & 
                                  (predicted_pfas['kmd_score'] > 0) & 
                                  (predicted_pfas['cf2_score'] == 0)])
    all_three = len(predicted_pfas[(predicted_pfas['cf2_score'] > 0) & 
                                   (predicted_pfas['fragment_score'] > 0) & 
                                   (predicted_pfas['kmd_score'] > 0)])
    
    labels = []
    sizes = []
    if kmd_only > 0:
        labels.append(f'KMD only ({kmd_only})')
        sizes.append(kmd_only)
    if cf2_only > 0:
        labels.append(f'CF₂ only ({cf2_only})')
        sizes.append(cf2_only)
    if frag_only > 0:
        labels.append(f'Fragment only ({frag_only})')
        sizes.append(frag_only)
    if cf2_kmd > 0:
        labels.append(f'CF₂+KMD ({cf2_kmd})')
        sizes.append(cf2_kmd)
    if cf2_frag > 0:
        labels.append(f'CF₂+Fragment ({cf2_frag})')
        sizes.append(cf2_frag)
    if frag_kmd > 0:
        labels.append(f'Fragment+KMD ({frag_kmd})')
        sizes.append(frag_kmd)
    if all_three > 0:
        labels.append(f'All three ({all_three})')
        sizes.append(all_three)
    
    colors = plt.cm.Set3(range(len(sizes)))
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Detection Method Combinations\n(Predicted PFAS)', fontsize=12, fontweight='bold')
    
    # Plot 4: Total score distribution
    ax4 = axes[1, 1]
    
    actual_pfas = df[df['is_PFAS'] == True]
    actual_non_pfas = df[df['is_PFAS'] == False]
    
    bins = np.arange(0, df['total_score'].max() + 2, 1)
    ax4.hist(actual_non_pfas['total_score'], bins=bins, alpha=0.6, 
            label=f'Non-PFAS (n={len(actual_non_pfas)})', color='gray', density=True)
    ax4.hist(actual_pfas['total_score'], bins=bins, alpha=0.7, 
            label=f'Actual PFAS (n={len(actual_pfas)})', color='red', density=True)
    
    ax4.axvline(x=5, color='blue', linestyle='--', linewidth=2, label='Classification threshold (5)')
    ax4.set_xlabel('Total Score', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Total Score Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved score contribution plot to {output_path}")
    plt.close()

def plot_kmd_performance_by_threshold(df, output_path):
    """Plot how performance metrics change with KMD threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    thresholds = np.linspace(0.01, 0.3, 30)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        # Recalculate predictions with different threshold
        df_temp = df.copy()
        df_temp['kmd_score_temp'] = (np.abs(df_temp['kmd']) <= thresh).astype(int) * 4
        df_temp['total_score_temp'] = df_temp['cf2_score'] + df_temp['fragment_score'] + df_temp['kmd_score_temp']
        df_temp['predicted_pfas_temp'] = df_temp['total_score_temp'] >= 5
        
        y_true = df_temp['is_PFAS'].astype(bool)
        y_pred = df_temp['predicted_pfas_temp'].astype(bool)
        
        # Calculate metrics
        tp = ((y_true == True) & (y_pred == True)).sum()
        fp = ((y_true == False) & (y_pred == True)).sum()
        fn = ((y_true == True) & (y_pred == False)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot 1: Metrics vs threshold
    ax1 = axes[0]
    ax1.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=4)
    ax1.plot(thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=4)
    ax1.plot(thresholds, f1_scores, '^-', label='F1 Score', linewidth=2, markersize=4)
    ax1.axvline(x=0.15, color='red', linestyle='--', linewidth=2, label='Current threshold (0.15)')
    
    ax1.set_xlabel('KMD Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance vs KMD Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    ax1.plot(optimal_thresh, optimal_f1, 'r*', markersize=15, 
            label=f'Optimal: {optimal_thresh:.3f} (F1={optimal_f1:.4f})')
    ax1.legend(fontsize=9)
    
    # Plot 2: Number of samples within threshold
    ax2 = axes[1]
    samples_within = [((np.abs(df['kmd']) <= t).sum()) for t in thresholds]
    pfas_within = [((np.abs(df[df['is_PFAS'] == True]['kmd']) <= t).sum()) for t in thresholds]
    
    ax2.plot(thresholds, samples_within, 'o-', label='Total samples', linewidth=2, markersize=4, color='blue')
    ax2.plot(thresholds, pfas_within, 's-', label='Actual PFAS', linewidth=2, markersize=4, color='red')
    ax2.axvline(x=0.15, color='green', linestyle='--', linewidth=2, label='Current threshold (0.15)')
    
    ax2.set_xlabel('KMD Threshold', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Samples Within KMD Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance vs threshold plot to {output_path}")
    plt.close()

def main():
    # Load results
    results_path = "results/pfas_predictions_with_kmd.tsv"
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    
    df = load_results(results_path)
    
    print("\nGenerating visualizations...")
    print("=" * 80)
    
    # Generate plots
    plot_kmd_vs_kendrick_mass(df, output_dir / "kmd_scatter.png")
    plot_kmd_distribution(df, output_dir / "kmd_distribution.png")
    plot_score_contributions(df, output_dir / "score_contributions.png")
    plot_kmd_performance_by_threshold(df, output_dir / "kmd_threshold_analysis.png")
    
    print("\n" + "=" * 80)
    print("All visualizations saved to results/plots/")
    print("=" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    pfas = df[df['is_PFAS'] == True]
    non_pfas = df[df['is_PFAS'] == False]
    
    print(f"Total samples: {len(df)}")
    print(f"Actual PFAS: {len(pfas)} ({len(pfas)/len(df)*100:.1f}%)")
    print(f"Actual non-PFAS: {len(non_pfas)} ({len(non_pfas)/len(df)*100:.1f}%)")
    print()
    print(f"Mean |KMD| for PFAS: {pfas['kmd'].abs().mean():.6f}")
    print(f"Mean |KMD| for non-PFAS: {non_pfas['kmd'].abs().mean():.6f}")
    print()
    print(f"PFAS within threshold (|KMD| <= 0.15): {(pfas['kmd'].abs() <= 0.15).sum()} ({(pfas['kmd'].abs() <= 0.15).sum()/len(pfas)*100:.1f}%)")
    print(f"Non-PFAS within threshold: {(non_pfas['kmd'].abs() <= 0.15).sum()} ({(non_pfas['kmd'].abs() <= 0.15).sum()/len(non_pfas)*100:.1f}%)")
    print()
    
    predicted_pfas = df[df['predicted_pfas'] == True]
    print(f"Predicted PFAS: {len(predicted_pfas)}")
    print(f"  - With KMD evidence: {(predicted_pfas['kmd_score'] > 0).sum()}")
    print(f"  - With CF2 evidence: {(predicted_pfas['cf2_score'] > 0).sum()}")
    print(f"  - With fragment evidence: {(predicted_pfas['fragment_score'] > 0).sum()}")

if __name__ == "__main__":
    main()
