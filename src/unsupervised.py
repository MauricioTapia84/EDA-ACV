from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA


def run_unsupervised(random_state: int = 42) -> None:
    root = Path(__file__).resolve().parents[1]
    x_train_path = root / "data" / "processed" / "X_train.csv"
    if not x_train_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/X_train.csv. Run preprocess phase first."
        )

    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(x_train_path)
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
    km_labels = kmeans.fit_predict(X_2d)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=km_labels, palette="viridis", s=20, legend=False)
    plt.title("PCA projection with KMeans clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(figure_dir / "pca_clusters.png")
    plt.close()

    agglomerative = AgglomerativeClustering(n_clusters=5)
    agg_labels = agglomerative.fit_predict(X_2d)
    dbscan = DBSCAN(eps=0.7, min_samples=10)
    db_labels = dbscan.fit_predict(X_2d)

    summary = pd.DataFrame(
        {
            "method": ["kmeans", "agglomerative", "dbscan"],
            "n_clusters_detected": [
                len(set(km_labels)),
                len(set(agg_labels)),
                len(set(db_labels)) - (1 if -1 in set(db_labels) else 0),
            ],
        }
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(data=summary, x="method", y="n_clusters_detected", hue="method", dodge=False, legend=False)
    plt.title("Clustering summary")
    plt.ylabel("Detected clusters")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(figure_dir / "clustering_summary.png")
    plt.close()

    print(f"Unsupervised analysis complete. Figures saved in {figure_dir}")


if __name__ == "__main__":
    run_unsupervised()