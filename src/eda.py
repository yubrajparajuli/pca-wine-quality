import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def plot_feature_distributions(X, output_path):
    """Plots histogram distribution of all features."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(X.columns):
        axes[i].hist(X[col], bins=30, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')

    axes[-1].set_visible(False)
    plt.suptitle('Distribution of All Wine Features',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(X, output_path):
    """Plots correlation heatmap of all features."""
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        X.corr(), annot=True, fmt='.2f',
        cmap='coolwarm', center=0, ax=ax,
        linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    ax.set_title('Feature Correlation Heatmap',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_quality_distribution(df, output_path):
    """Plots original and binary quality distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#e74c3c' if q <= 5 else '#2ecc71'
              for q in sorted(df['quality'].unique())]
    quality_counts = df['quality'].value_counts().sort_index()

    bars = axes[0].bar(quality_counts.index, quality_counts.values,
                       color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_xlabel('Quality Score', fontweight='bold')
    axes[0].set_ylabel('Number of Wines', fontweight='bold')
    axes[0].set_title('Original Quality Distribution', fontweight='bold')
    for bar, count in zip(bars, quality_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5,
                     str(count), ha='center', fontweight='bold')
    red_patch   = mpatches.Patch(color='#e74c3c', label='Bad Wine (≤5)')
    green_patch = mpatches.Patch(color='#2ecc71', label='Good Wine (≥6)')
    axes[0].legend(handles=[red_patch, green_patch])

    binary_counts = df['target'].value_counts().sort_index()
    bars2 = axes[1].bar(['Bad Wine (0)', 'Good Wine (1)'],
                        binary_counts.values,
                        color=['#e74c3c', '#2ecc71'],
                        edgecolor='white', linewidth=1.5, width=0.5)
    axes[1].set_xlabel('Wine Class', fontweight='bold')
    axes[1].set_ylabel('Number of Wines', fontweight='bold')
    axes[1].set_title('Binary Target Distribution', fontweight='bold')
    for bar, count in zip(bars2, binary_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5,
                     str(count), ha='center', fontweight='bold')

    plt.suptitle('Wine Quality — Original vs Binary Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots(X, y, output_path):
    """Plots boxplots of each feature split by target class."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(X.columns):
        bad_wine  = X[col][y == 0]
        good_wine = X[col][y == 1]
        axes[i].boxplot(
            [bad_wine, good_wine],
            labels=['Bad Wine', 'Good Wine'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='navy'),
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markersize=3,
                           markerfacecolor='gray')
        )
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Value')

    axes[-1].set_visible(False)
    plt.suptitle('Feature Distribution: Bad Wine vs Good Wine',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()