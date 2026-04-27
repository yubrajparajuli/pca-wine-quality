import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from src.data_loader    import load_data
from src.eda            import (plot_feature_distributions,
                                plot_correlation_heatmap,
                                plot_quality_distribution,
                                plot_boxplots)
from src.preprocessing  import standardize
from src.pca_scratch    import run_pca_scratch
from src.pca_sklearn    import run_pca_sklearn, verify_pca
from src.visualization  import (plot_scree,
                                plot_2d_scatter,
                                plot_biplot,
                                plot_final_comparison)
from src.model          import train_evaluate


def main():

    print("=" * 55)
    print("   PCA ON WINE QUALITY DATASET")
    print("=" * 55)

    # --- Load Data ---
    print("\n[1/8] Loading data...")
    X, y, df = load_data('data/WineQT.csv')
    print(f"      X shape: {X.shape} | y shape: {y.shape}")

    # --- EDA ---
    print("\n[2/8] Running EDA...")
    plot_feature_distributions(X, 'outputs/01_feature_distributions.png')
    plot_correlation_heatmap(X, 'outputs/02_correlation_heatmap.png')
    plot_quality_distribution(df, 'outputs/03_quality_distribution.png')
    plot_boxplots(X, y, 'outputs/04_boxplot_features_vs_target.png')
    print("      EDA plots saved to outputs/")

    # --- Standardize ---
    print("\n[3/8] Standardizing data...")
    X_scaled, scaler = standardize(X)
    print("      Standardization complete")

    # --- PCA from Scratch ---
    print("\n[4/8] Running PCA from scratch...")
    scratch = run_pca_scratch(X_scaled, variance_threshold=0.80)
    n_components = scratch["n_components"]
    print(f"      Components selected: {n_components}")
    print(f"      Variance retained:   "
          f"{scratch['cumulative_variance'][n_components-1]*100:.2f}%")

    # --- PCA sklearn ---
    print("\n[5/8] Running sklearn PCA...")
    X_pca_sklearn, pca = run_pca_sklearn(X_scaled, n_components)
    verify_pca(pca, scratch)

    # --- PCA Visualizations ---
    print("\n[6/8] Creating PCA visualizations...")
    plot_scree(
        scratch["eigenvalues"],
        scratch["explained_variance_ratio"],
        scratch["cumulative_variance"],
        n_components,
        'outputs/05_scree_plot.png'
    )
    plot_2d_scatter(X_pca_sklearn, y, pca,
                    'outputs/06_pca_2d_scatter.png')
    plot_biplot(X_pca_sklearn, y, pca,
                list(X.columns),
                'outputs/07_biplot.png')
    print("      PCA plots saved to outputs/")

    # --- Model Before PCA ---
    print("\n[7/8] Training model BEFORE PCA...")
    results_full = train_evaluate(X_scaled, y, label="BEFORE PCA")

    # --- Model After PCA ---
    print("\n[8/8] Training model AFTER PCA...")
    results_pca = train_evaluate(X_pca_sklearn, y, label="AFTER PCA")

    # --- Final Comparison ---
    plot_final_comparison(
        results_full["accuracy"],
        results_pca["accuracy"],
        results_full["duration"],
        results_pca["duration"],
        n_components,
        scratch["cumulative_variance"],
        'outputs/08_final_comparison.png'
    )

    # --- Summary ---
    print("\n" + "=" * 55)
    print("   FINAL SUMMARY")
    print("=" * 55)
    print(f"  Features reduced:    11 → {n_components} components")
    print(f"  Variance retained:   "
          f"{scratch['cumulative_variance'][n_components-1]*100:.2f}%")
    print(f"  Accuracy before PCA: {results_full['accuracy']*100:.2f}%")
    print(f"  Accuracy after PCA:  {results_pca['accuracy']*100:.2f}%")
    print(f"  Accuracy drop:       "
          f"{(results_full['accuracy']-results_pca['accuracy'])*100:.2f}%")
    print(f"  Speed improvement:   "
          f"{results_full['duration']/results_pca['duration']:.0f}x faster")
    print("\n  All outputs saved to outputs/")
    print("=" * 55)


if __name__ == "__main__":
    main()