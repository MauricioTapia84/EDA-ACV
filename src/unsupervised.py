"""Analisis no supervisado coherente con el flujo actual del proyecto ACV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .data_preprocessing import build_unsupervised_matrix, load_raw_dataset


def run_unsupervised(
    input_path: str | Path = "data/raw/healthcare-dataset-stroke-data.csv",
    output_dir: str | Path = "results/plots",
) -> pd.DataFrame:
    """Ejecuta PCA y clustering guardando la evidencia en ``results/plots``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Iniciando analisis no supervisado...")
    df_raw = load_raw_dataset(input_path)
    _, X_matrix, _ = build_unsupervised_matrix(df_raw)

    print("Calculando PCA para visualizacion...")
    pca_full = PCA(n_components=0.95, random_state=42)
    X_pca_full = pca_full.fit_transform(X_matrix)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
    plt.axhline(y=0.95, color="r", linestyle="--")
    plt.title("Varianza explicada acumulada")
    plt.xlabel("Numero de componentes")
    plt.ylabel("Varianza acumulada")
    plt.savefig(output_path / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_matrix)

    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)
    sample_size = min(2000, len(X_pca_full))
    X_sample = X_pca_full[:sample_size]

    print("Calculando metodo del codo y silueta para K-Means...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_sample)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_sample, labels))

    results = pd.DataFrame(
        {
            "k": list(k_range),
            "inertia": inertia,
            "silhouette": silhouette_scores,
        }
    )

    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["inertia"], marker="o", linewidth=2)
    plt.title("Metodo del codo")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Inercia")
    plt.savefig(output_path / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["silhouette"], marker="o", linewidth=2, color="teal")
    plt.title("Indice de silueta por numero de clusters")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Puntuacion de silueta")
    plt.savefig(output_path / "kmeans_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_k = int(results.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"])

    print("Generando visualizaciones de clusters...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_pca_2d)
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=km.labels_, palette="viridis", ax=axes[0], legend=False)
    axes[0].set_title(f"K-Means (k={best_k})")

    sample_2d = X_pca_2d[:sample_size]
    agg = AgglomerativeClustering(n_clusters=best_k).fit(sample_2d)
    sns.scatterplot(x=sample_2d[:, 0], y=sample_2d[:, 1], hue=agg.labels_, palette="viridis", ax=axes[1], legend=False)
    axes[1].set_title("Agrupamiento jerarquico")

    db = DBSCAN(eps=0.5, min_samples=5).fit(sample_2d)
    sns.scatterplot(x=sample_2d[:, 0], y=sample_2d[:, 1], hue=db.labels_, palette="viridis", ax=axes[2], legend=False)
    axes[2].set_title("DBSCAN")

    plt.tight_layout()
    plt.savefig(output_path / "clusters_2d_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Analisis no supervisado finalizado. Graficos en {output_path}/")
    return results
