import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # Biblioteka umap-learn

# Importy zgodnie z Twoją nową strukturą (plik visualization.py w folderze utils)
from utils.preprocessing.preprocess_yellow import preprocess_yellow, sample_data, scale_data
from utils.visualization import plot_altair


def main():

    data_path = "data/raw/yellow_tripdata_2026-03.parquet"
    lookup_path = "data/taxi_zone_lookup.csv"
    output_dir = "plots"
    n_samples = 20000

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Wczytywanie i przetwarzanie danych...")
    df_raw = pd.read_parquet(data_path)
    df_processed = preprocess_yellow(df_raw, lookup_path=lookup_path)
    df_sample = sample_data(df_processed, n_samples=n_samples)
    X_scaled, _ = scale_data(df_sample)

    metadata_cols = [
        'trip_distance', 'fare_amount', 'duration',
        'pickup_hour', 'PU_Borough', 'DO_Borough'
    ]

    # --- METODA 1: PCA ---
    print("Obliczanie PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df_sample.index)
    df_plot_pca = df_pca.join(df_sample[metadata_cols])

    print("Zapisywanie wizualizacji PCA...")
    chart_pca = plot_altair(df_plot_pca, x_col='PC1', y_col='PC2', title="Analiza PCA")
    chart_pca.save(os.path.join(output_dir, "pca_visualization.html"))

    # --- METODA 2: t-SNE ---
    print("Obliczanie t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)

    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'], index=df_sample.index)
    df_plot_tsne = df_tsne.join(df_sample[metadata_cols])

    print("Zapisywanie wizualizacji t-SNE...")
    chart_tsne = plot_altair(df_plot_tsne, x_col='TSNE1', y_col='TSNE2', title="Analiza t-SNE")
    chart_tsne.save(os.path.join(output_dir, "tsne_visualization.html"))

    # --- METODA 3: UMAP ---
    print("Obliczanie UMAP...")
    # Parametry n_neighbors i min_dist kontrolują balans między lokalną a globalną strukturą
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=df_sample.index)
    df_plot_umap = df_umap.join(df_sample[metadata_cols])

    print("Zapisywanie wizualizacji UMAP...")
    chart_umap = plot_altair(df_plot_umap, x_col='UMAP1', y_col='UMAP2', title="Analiza UMAP")
    chart_umap.save(os.path.join(output_dir, "umap_visualization.html"))

    print(f"\nSukces! Wygenerowano 3 pliki HTML w folderze: {output_dir}")


if __name__ == "__main__":
    main()