# PCA on Wine Quality Dataset

A complete implementation of Principal Component Analysis (PCA) built from scratch using NumPy, verified with sklearn, and applied to the Wine Quality dataset with a Logistic Regression comparison before and after dimensionality reduction.

---

## Project Structure

pca-wine-quality/
├── data/
│ └── WineQT.csv
├── notebooks/
│ └── pca_wine_quality.ipynb
├── outputs/
│ ├── 01_feature_distributions.png
│ ├── 02_correlation_heatmap.png
│ ├── 03_quality_distribution.png
│ ├── 04_boxplot_features_vs_target.png
│ ├── 05_scree_plot.png
│ ├── 06_pca_2d_scatter.png
│ ├── 07_biplot.png
│ └── 08_final_comparison.png
├── src/
│ ├── data_loader.py
│ ├── eda.py
│ ├── preprocessing.py
│ ├── pca_scratch.py
│ ├── pca_sklearn.py
│ ├── visualization.py
│ └── model.py
├── main.py
├── requirements.txt
└── README.md

---

## What This Project Covers

- Exploratory Data Analysis with 4 detailed visualizations
- Feature standardization using StandardScaler
- PCA built from scratch using NumPy only
  - Data centering
  - Covariance matrix computation
  - Eigenvector and eigenvalue decomposition
  - Component selection based on variance threshold
  - Data projection
- PCA verification against sklearn — zero difference confirmed
- Logistic Regression trained before and after PCA
- Full accuracy, speed and dimensionality comparison

---

## Key Results

| Metric            | Before PCA | After PCA  |
| ----------------- | ---------- | ---------- |
| Features          | 11         | 6          |
| Variance retained | 100%       | 85.85%     |
| Accuracy          | 77.73%     | 77.29%     |
| Accuracy drop     | —          | 0.44%      |
| Training speed    | baseline   | 18x faster |

Reducing 11 features to 6 principal components retained 85.85% of variance while dropping accuracy by only 0.44% and improving training speed by 18x.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yubrajparajuli/pca-wine-quality.git
cd pca-wine-quality

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel

# Run full pipeline
uv run python main.py
```

---

## Dataset

Wine Quality Dataset from Kaggle — 1143 red wine samples with 11 chemical features and a quality score from 3 to 8.

Source: [Kaggle — Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

---

## Tech Stack

- Python 3.11+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- UV (package manager)
