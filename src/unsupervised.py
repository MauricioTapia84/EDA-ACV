import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def run_unsupervised():
    print("Iniciando Análisis No Supervisado...")
    os.makedirs("outputs/figures", exist_ok=True)
    
   
    X = pd.read_csv("data/processed/X_train.csv").values

    # 1. Calculamos el PCA 
    print("Calculando PCA para visualización...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Gráfico de Varianza Explicada
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title("Varianza Explicada Acumulada")
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza Acumulada")
    plt.savefig("outputs/figures/pca_variance.png")
    plt.close()

    # Reducimos a 2D para visualización
    X_pca_2d = X[:, :2] 

    # 2. Método del Codo y Silueta (K-Means)
    inertia = []
    sil_scores = []
    K_range = range(2, 11)
    
    # Usamos una muestra pequeña para que silueta no demore horas
    X_sample = X[:2000] 
    
    print("Calculando Codo y Silueta para K-Means...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_sample, kmeans.labels_))
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(K_range, inertia, 'b-o')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inercia', color='b')
    ax2 = ax1.twinx()
    ax2.plot(K_range, sil_scores, 'r-o')
    ax2.set_ylabel('Puntuación Silueta', color='r')
    plt.title("Método del Codo y Silueta")
    plt.savefig("outputs/figures/kmeans_elbow_silhouette.png")
    plt.close()

    # 3. Clustering Visualizations
    print("Generando visualizaciones de clusters...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # K-Means (Asumiendo 5 clusters por ejemplo visual)
    km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_pca_2d)
    sns.scatterplot(x=X_pca_2d[:,0], y=X_pca_2d[:,1], hue=km.labels_, palette="viridis", ax=axes[0])
    axes[0].set_title("K-Means (k=5)")
    
    # Jerárquico
    agg = AgglomerativeClustering(n_clusters=5).fit(X_pca_2d[:2000]) # Muestra para evitar OOM
    sns.scatterplot(x=X_pca_2d[:2000,0], y=X_pca_2d[:2000,1], hue=agg.labels_, palette="viridis", ax=axes[1])
    axes[1].set_title("Hierarchical Clustering")
    
    # DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5).fit(X_pca_2d)
    sns.scatterplot(x=X_pca_2d[:,0], y=X_pca_2d[:,1], hue=db.labels_, palette="viridis", ax=axes[2])
    axes[2].set_title("DBSCAN")
    
    plt.tight_layout()
    plt.savefig("outputs/figures/clusters_2d_comparison.png")
    plt.close()
    print("Análisis No Supervisado Finalizado. Gráficos en outputs/figures/")