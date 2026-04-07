import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from clustering import run_clustering
import os

def visualize_clusters():

    numeric_df, kmeans = run_clustering()

    # PCA for visualization
    X = numeric_df.drop(columns=["cluster", "cluster_name"], errors="ignore")
    X = X.select_dtypes(include=["int64","float64"])

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X)

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(
        pca_components[:,0],
        pca_components[:,1],
        c=numeric_df["cluster"],
        cmap="viridis"
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Music Clusters Visualization")
    plt.colorbar(label="Cluster")
    plt.savefig("outputs/cluster_visualization.png")
    plt.show()

    
    
    cluster_summary = numeric_df.groupby("cluster").mean(numeric_only=True)

    plt.figure(figsize=(12,6))
    sns.heatmap(cluster_summary, cmap="coolwarm")
    plt.title("Cluster Feature Comparison")
    plt.savefig("outputs/heatmap.png")
    plt.show()


if __name__ == "__main__":
    visualize_clusters()