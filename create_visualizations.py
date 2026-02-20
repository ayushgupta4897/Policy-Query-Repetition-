#!/usr/bin/env python3
"""
Generate aesthetic visualizations for PQR experiment results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# Color palette
COLORS = {
    'baseline': '#e74c3c',  # Red
    'pqr': '#27ae60',       # Green
    'accent': '#3498db',    # Blue
    'neutral': '#95a5a6',   # Gray
}

# Results data
RESULTS = {
    'llama-3.3-70b': {'baseline': 38.5, 'pqr': 0.0, 'improvement': 100.0},
    'qwen-2.5-72b': {'baseline': 32.3, 'pqr': 7.7, 'improvement': 76.2},
    'gpt-4o': {'baseline': 20.0, 'pqr': 7.7, 'improvement': 61.5},
    'gpt-4.1': {'baseline': 15.4, 'pqr': 7.7, 'improvement': 50.0},
    'gemini-2.0-flash': {'baseline': 49.2, 'pqr': 35.4, 'improvement': 28.1},
    'gpt-3.5-turbo': {'baseline': 43.1, 'pqr': 35.4, 'improvement': 17.9},
}

OUTPUT_DIR = 'results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_comparison_bar_chart():
    """Create side-by-side bar chart comparing baseline vs PQR."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(RESULTS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    baseline_vals = [RESULTS[m]['baseline'] for m in models]
    pqr_vals = [RESULTS[m]['pqr'] for m in models]
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (P+Q)', 
                   color=COLORS['baseline'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, pqr_vals, width, label='PQR (P+Q+P+Q)', 
                   color=COLORS['pqr'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontweight='bold')
    ax.set_title('Policy Violation Rates: Baseline vs PQR', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=9)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/comparison_bar_chart.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/comparison_bar_chart.png")


def create_improvement_chart():
    """Create horizontal bar chart showing improvement percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by improvement
    sorted_models = sorted(RESULTS.keys(), key=lambda m: RESULTS[m]['improvement'], reverse=True)
    improvements = [RESULTS[m]['improvement'] for m in sorted_models]
    
    y = np.arange(len(sorted_models))
    
    # Color bars based on target achievement
    colors = [COLORS['pqr'] if imp >= 30 else COLORS['neutral'] for imp in improvements]
    
    bars = ax.barh(y, improvements, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, height=0.6)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.annotate(f'{imp:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold',
                    color=COLORS['pqr'] if imp >= 30 else COLORS['neutral'])
    
    # Add target line
    ax.axvline(x=30, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(31, len(sorted_models) - 0.5, '30% Target', color=COLORS['accent'], 
            fontsize=10, fontstyle='italic')
    
    ax.set_xlabel('Improvement (%)', fontweight='bold')
    ax.set_title('PQR Improvement by Model', fontweight='bold', fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace('-', ' ').title() for m in sorted_models], fontsize=10)
    ax.set_xlim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    target_met = mpatches.Patch(color=COLORS['pqr'], label='Target Met (≥30%)')
    target_missed = mpatches.Patch(color=COLORS['neutral'], label='Below Target (<30%)')
    ax.legend(handles=[target_met, target_missed], loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/improvement_chart.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/improvement_chart.png")


def create_before_after_chart():
    """Create a connected dot plot showing before/after."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sorted_models = sorted(RESULTS.keys(), key=lambda m: RESULTS[m]['improvement'], reverse=True)
    
    y_positions = np.arange(len(sorted_models))
    
    for i, model in enumerate(sorted_models):
        baseline = RESULTS[model]['baseline']
        pqr = RESULTS[model]['pqr']
        
        # Connect line
        ax.plot([baseline, pqr], [i, i], color=COLORS['accent'], linewidth=2, alpha=0.6, zorder=1)
        
        # Arrow showing direction
        ax.annotate('', xy=(pqr, i), xytext=(baseline, i),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
        
        # Baseline point
        ax.scatter(baseline, i, s=120, color=COLORS['baseline'], zorder=2, edgecolor='white', linewidth=2)
        
        # PQR point
        ax.scatter(pqr, i, s=120, color=COLORS['pqr'], zorder=2, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Violation Rate (%)', fontweight='bold')
    ax.set_title('Violation Rate Reduction: Baseline → PQR', fontweight='bold', fontsize=14)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m.replace('-', ' ').title() for m in sorted_models], fontsize=10)
    ax.set_xlim(-5, 55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    baseline_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'], 
                               markersize=10, label='Baseline')
    pqr_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pqr'], 
                          markersize=10, label='PQR')
    ax.legend(handles=[baseline_dot, pqr_dot], loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/before_after_chart.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/before_after_chart.png")


def create_summary_infographic():
    """Create a summary infographic."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Overall improvement
    ax1 = axes[0]
    ax1.pie([52.7, 47.3], labels=['Reduction', 'Remaining'], 
            colors=[COLORS['pqr'], COLORS['neutral']], autopct='%1.1f%%',
            startangle=90, explode=[0.05, 0])
    ax1.set_title('Overall Improvement', fontweight='bold', fontsize=12)
    
    # Models exceeding target
    ax2 = axes[1]
    exceeded = sum(1 for m in RESULTS.values() if m['improvement'] >= 30)
    total = len(RESULTS)
    ax2.pie([exceeded, total - exceeded], labels=['≥30%', '<30%'],
            colors=[COLORS['pqr'], COLORS['baseline']], autopct=lambda pct: f'{int(pct/100*total)}/{total}',
            startangle=90, explode=[0.05, 0])
    ax2.set_title('Models Meeting Target', fontweight='bold', fontsize=12)
    
    # Best performer
    ax3 = axes[2]
    ax3.text(0.5, 0.7, 'Best Performer', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.5, 'Llama 3.3 70B', ha='center', va='center', fontsize=16, 
             fontweight='bold', color=COLORS['pqr'])
    ax3.text(0.5, 0.3, '100% Improvement', ha='center', va='center', fontsize=14,
             color=COLORS['accent'])
    ax3.axis('off')
    
    plt.suptitle('PQR Experiment Summary', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_infographic.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/summary_infographic.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    create_comparison_bar_chart()
    create_improvement_chart()
    create_before_after_chart()
    create_summary_infographic()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
