import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess

def run_clustering():
    numeric_df, scaled_data = load_and_preprocess()

    inertia = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1,11), inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.savefig("outputs/elbow_plot.png")
    plt.show()

    kmeans = KMeans(n_clusters=4, random_state=42)

    clusters = kmeans.fit_predict(scaled_data)

    sil_score = silhouette_score(scaled_data, clusters)
    print("\nSilhouette Score:", sil_score)

    numeric_df["cluster"] = clusters

    cluster_labels = {
    0: "Chill Songs",
    1: "Party Tracks",
    2: "Melody Songs",
    3: "Energetic Songs"
}
    numeric_df["music_type"] = numeric_df["cluster"].map(cluster_labels)
    

    cluster_summary = numeric_df.groupby("cluster").mean(numeric_only=True)

    print("\nCluster Summary:\n")
    print(cluster_summary)    

    cluster_summary.to_csv("cluster_summary.csv")
    numeric_df.to_csv("clustered_music_dataset.csv", index=False)
    
    

    print("\nCluster summary saved as cluster_summary.csv")

    return numeric_df, kmeans

if __name__ == "__main__":
    run_clustering()