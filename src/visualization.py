import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_scree(eigenvalues, explained_variance_ratio,
               cumulative_variance, n_components, output_path):
    """Plots individual and cumulative explained variance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, len(eigenvalues)+1),
                explained_variance_ratio * 100,
                color='steelblue', edgecolor='white', alpha=0.8)
    for i, var in enumerate(explained_variance_ratio):
        axes[0].text(i+1, var*100 + 0.3,
                     f'{var*100:.1f}%',
                     ha='center', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Principal Component', fontweight='bold')
    axes[0].set_ylabel('Explained Variance (%)', fontweight='bold')
    axes[0].set_title('Individual Explained Variance', fontweight='bold')
    axes[0].set_xticks(range(1, len(eigenvalues)+1))

    axes[1].plot(range(1, len(eigenvalues)+1),
                 cumulative_variance * 100,
                 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[1].axhline(y=80, color='red', linestyle='--',
                    alpha=0.7, label='80% threshold')
    axes[1].axvline(x=n_components, color='green', linestyle='--',
                    alpha=0.7, label=f'{n_components} components selected')
    for i, var in enumerate(cumulative_variance):
        axes[1].text(i+1, var*100 + 1.5,
                     f'{var*100:.1f}%',
                     ha='center', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Number of Components', fontweight='bold')
    axes[1].set_ylabel('Cumulative Variance (%)', fontweight='bold')
    axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
    axes[1].set_xticks(range(1, len(eigenvalues)+1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('PCA Scree Plot — Wine Quality Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_2d_scatter(X_pca, y, pca, output_path):
    """Plots 2D PCA scatter of good vs bad wine."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {0: '#e74c3c', 1: '#2ecc71'}
    labels = {0: 'Bad Wine',  1: 'Good Wine'}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[cls], label=labels[cls],
            alpha=0.6, edgecolors='white',
            linewidth=0.5, s=60
        )

    ax.set_xlabel(
        f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
        fontweight='bold')
    ax.set_ylabel(
        f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
        fontweight='bold')
    ax.set_title('PCA — 2D Projection of Wine Quality\n'
                 'Red = Bad Wine | Green = Good Wine',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_biplot(X_pca, y, pca, feature_names, output_path):
    """Plots PCA biplot with feature arrows and data points."""
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = {0: '#e74c3c', 1: '#2ecc71'}
    labels = {0: 'Bad Wine',  1: 'Good Wine'}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[cls], alpha=0.3, s=30,
            edgecolors='white', linewidth=0.3,
            label=labels[cls]
        )

    components = pca.components_
    scale = 3.5
    for i, feature in enumerate(feature_names):
        ax.annotate('',
                    xy=(components[0, i]*scale,
                        components[1, i]*scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->',
                                   color='black', lw=1.5))
        ax.text(components[0, i]*scale*1.18,
                components[1, i]*scale*1.18,
                feature, fontsize=9, fontweight='bold',
                color='darkblue', ha='center', va='center')

    ax.set_xlabel(
        f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
        fontweight='bold')
    ax.set_ylabel(
        f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
        fontweight='bold')
    ax.set_title('PCA Biplot — Wine Features & Samples',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_comparison(accuracy_full, accuracy_pca,
                          time_full, time_pca,
                          n_components, cumulative_variance,
                          output_path):
    """Plots final comparison of before vs after PCA."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    bars1 = axes[0].bar(
        ['Before PCA\n(11 features)', f'After PCA\n({n_components} components)'],
        [accuracy_full*100, accuracy_pca*100],
        color=['#3498db', '#e67e22'],
        edgecolor='white', linewidth=1.5, width=0.5
    )
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].set_ylim([70, 82])
    for bar, val in zip(bars1, [accuracy_full*100, accuracy_pca*100]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     f'{val:.2f}%', ha='center',
                     fontweight='bold', fontsize=12)

    bars2 = axes[1].bar(
        ['Before PCA\n(11 features)', f'After PCA\n({n_components} components)'],
        [time_full, time_pca],
        color=['#3498db', '#e67e22'],
        edgecolor='white', linewidth=1.5, width=0.5
    )
    axes[1].set_ylabel('Training Time (seconds)', fontweight='bold')
    axes[1].set_title('Training Speed', fontweight='bold')
    for bar, val in zip(bars2, [time_full, time_pca]):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f'{val:.5f}s', ha='center',
                     fontweight='bold', fontsize=11)

    bars3 = axes[2].bar(
        ['Original\nFeatures', 'PCA\nComponents'],
        [11, n_components],
        color=['#3498db', '#e67e22'],
        edgecolor='white', linewidth=1.5, width=0.5
    )
    axes[2].set_ylabel('Number of Features', fontweight='bold')
    axes[2].set_title('Dimensionality Reduction', fontweight='bold')
    axes[2].text(1, n_components + 0.2,
                 f'{cumulative_variance[n_components-1]*100:.1f}%\nvariance retained',
                 ha='center', fontweight='bold',
                 fontsize=11, color='darkgreen')
    for bar, val in zip(bars3, [11, n_components]):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     str(val), ha='center',
                     fontweight='bold', fontsize=13)

    plt.suptitle('PCA Impact on Model Performance — Wine Quality Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()