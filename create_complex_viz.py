#!/usr/bin/env python3
"""
Generate visualizations for complex policy experiment results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

COLORS = {
    'baseline': '#e74c3c',
    'pqr': '#27ae60',
    'accent': '#3498db',
    'neutral': '#95a5a6',
}

# Complex policy results
COMPLEX_RESULTS = {
    'gpt-4o': {'baseline': 16.4, 'pqr': 0.0, 'improvement': 100.0},
    'gpt-4.1': {'baseline': 25.5, 'pqr': 5.5, 'improvement': 78.6},
    'qwen-2.5-72b': {'baseline': 63.6, 'pqr': 29.1, 'improvement': 54.3},
    'gpt-3.5-turbo': {'baseline': 18.2, 'pqr': 9.1, 'improvement': 50.0},
    'llama-3.3-70b': {'baseline': 21.8, 'pqr': 18.2, 'improvement': 16.7},
    'gemini-2.0-flash': {'baseline': 34.5, 'pqr': 36.4, 'improvement': -5.3},
}

# Simple policy results for comparison
SIMPLE_RESULTS = {
    'gpt-4o': {'baseline': 20.0, 'pqr': 7.7, 'improvement': 61.5},
    'gpt-4.1': {'baseline': 15.4, 'pqr': 7.7, 'improvement': 50.0},
    'qwen-2.5-72b': {'baseline': 32.3, 'pqr': 7.7, 'improvement': 76.2},
    'gpt-3.5-turbo': {'baseline': 43.1, 'pqr': 35.4, 'improvement': 17.9},
    'llama-3.3-70b': {'baseline': 38.5, 'pqr': 0.0, 'improvement': 100.0},
    'gemini-2.0-flash': {'baseline': 49.2, 'pqr': 35.4, 'improvement': 28.1},
}

OUTPUT_DIR = 'results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_comparison_chart():
    """Compare simple vs complex policy results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(COMPLEX_RESULTS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Simple policy
    ax1 = axes[0]
    baseline_vals = [SIMPLE_RESULTS[m]['baseline'] for m in models]
    pqr_vals = [SIMPLE_RESULTS[m]['pqr'] for m in models]
    
    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color=COLORS['baseline'], alpha=0.85)
    ax1.bar(x + width/2, pqr_vals, width, label='PQR', color=COLORS['pqr'], alpha=0.85)
    ax1.set_title('Simple Policy (~500 chars)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split('-')[0] for m in models], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 70)
    
    # Complex policy
    ax2 = axes[1]
    baseline_vals = [COMPLEX_RESULTS[m]['baseline'] for m in models]
    pqr_vals = [COMPLEX_RESULTS[m]['pqr'] for m in models]
    
    ax2.bar(x - width/2, baseline_vals, width, label='Baseline', color=COLORS['baseline'], alpha=0.85)
    ax2.bar(x + width/2, pqr_vals, width, label='PQR', color=COLORS['pqr'], alpha=0.85)
    ax2.set_title('Complex Policy (MEGA ~8000 chars)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Violation Rate (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split('-')[0] for m in models], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 70)
    
    plt.suptitle('Simple vs Complex Policy: Baseline vs PQR', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/simple_vs_complex.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/simple_vs_complex.png")


def create_improvement_comparison():
    """Compare improvement rates between simple and complex policies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(COMPLEX_RESULTS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    simple_imp = [SIMPLE_RESULTS[m]['improvement'] for m in models]
    complex_imp = [COMPLEX_RESULTS[m]['improvement'] for m in models]
    
    bars1 = ax.bar(x - width/2, simple_imp, width, label='Simple Policy', color=COLORS['accent'], alpha=0.85)
    bars2 = ax.bar(x + width/2, complex_imp, width, label='Complex Policy (MEGA)', color='#9b59b6', alpha=0.85)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, max(0, height)),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        y_pos = max(0, height) if height >= 0 else height
        va = 'bottom' if height >= 0 else 'top'
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 3 if height >= 0 else -3), textcoords="offset points",
                    ha='center', va=va, fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('PQR Improvement: Simple vs Complex Policy', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(-20, 120)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/improvement_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/improvement_comparison.png")


def create_summary_stats():
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Simple policy overall
    ax1 = axes[0]
    simple_overall = 52.7
    ax1.pie([simple_overall, 100-simple_overall], 
            labels=['Improvement', 'Remaining'],
            colors=[COLORS['pqr'], COLORS['neutral']],
            autopct='%1.1f%%', startangle=90, explode=[0.05, 0])
    ax1.set_title('Simple Policy\nOverall Improvement', fontweight='bold')
    
    # Complex policy overall
    ax2 = axes[1]
    complex_overall = 45.5
    ax2.pie([complex_overall, 100-complex_overall],
            labels=['Improvement', 'Remaining'],
            colors=['#9b59b6', COLORS['neutral']],
            autopct='%1.1f%%', startangle=90, explode=[0.05, 0])
    ax2.set_title('Complex Policy\nOverall Improvement', fontweight='bold')
    
    # Best performer
    ax3 = axes[2]
    ax3.text(0.5, 0.75, 'Best Complex Policy Result', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(0.5, 0.5, 'GPT-4o', ha='center', va='center', fontsize=18, fontweight='bold', color='#9b59b6')
    ax3.text(0.5, 0.25, '100% Improvement\n(16.4% → 0.0%)', ha='center', va='center', fontsize=12, color=COLORS['accent'])
    ax3.axis('off')
    
    plt.suptitle('PQR Experiment Summary', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_stats.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/summary_stats.png")


if __name__ == "__main__":
    print("Generating complex policy visualizations...")
    create_comparison_chart()
    create_improvement_comparison()
    create_summary_stats()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
