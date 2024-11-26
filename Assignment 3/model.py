# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set(style='whitegrid')

# Task 1: Load and Explore the Dataset
# Load the data
headlines = pd.read_csv("headlines.csv")
tfidf_features = np.load("tfidf_features.npy")

# Check the shapes
print("Shape of headlines dataset:", headlines.shape)
print("Shape of TF-IDF features:", tfidf_features.shape)

# Display a few headlines
print("Sample headlines:\n", headlines.head())

# Task 2: Perform K-Means Clustering
# Determine the optimal number of clusters using the Elbow Method and Silhouette Score
inertia = []
sil_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(tfidf_features)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(tfidf_features, kmeans.labels_))

# Plot the Elbow Curve
plt.figure(figsize=(12, 6))
plt.plot(k_values, inertia, marker='o', label='Inertia')
plt.title('Elbow Curve for K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.grid()
plt.show()

# Additional visualization to highlight the elbow point
plt.figure(figsize=(12, 6))
plt.plot(k_values, inertia, marker='o', label='Inertia')
plt.axvline(x=5, color='red', linestyle='--', label='Optimal k (Elbow Point)')
plt.title('Elbow Method Highlight')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.grid()
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(12, 6))
plt.plot(k_values, sil_scores, marker='o', color='green', label='Silhouette Score')
plt.title('Silhouette Scores for K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid()
plt.show()

# Choose optimal k (based on the elbow curve and silhouette score)
optimal_k = 5  # Replace this with the observed optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
kmeans.fit(tfidf_features)

# Assign headlines to clusters
headlines['Cluster'] = kmeans.labels_

# Analyze contents of three clusters
for cluster in range(3):  # Analyze first three clusters
    print(f"\nCluster {cluster}:")
    print(headlines[headlines['Cluster'] == cluster].head(10)['headline'].values)

# Task 3: Perform Hierarchical Clustering on a Subset
# Select a random subset of 1,000 headlines
subset_indices = np.random.choice(range(len(headlines)), size=1000, replace=False)
subset_tfidf = tfidf_features[subset_indices]
subset_headlines = headlines.iloc[subset_indices]

# Perform Hierarchical Clustering
linked = linkage(subset_tfidf, method='ward')

# Plot dendrogram
plt.figure(figsize=(16, 8))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Headlines')
plt.ylabel('Distance')
plt.show()

# Assign clusters based on dendrogram
hierarchical_clusters = fcluster(linked, t=5, criterion='maxclust')
subset_headlines['Hierarchical_Cluster'] = hierarchical_clusters

# Task 4: Compare and Summarize
# Compare themes in K-Means and Hierarchical Clustering
print("\nHierarchical Clustering Themes:")
for cluster in range(1, 4):  # Analyze first three clusters
    print(f"\nCluster {cluster}:")
    print(subset_headlines[subset_headlines['Hierarchical_Cluster'] == cluster].head(10)['headline'].values)

# Optional Task: Clustering with PCA-Reduced Features
# Reduce TF-IDF to 50 dimensions
pca = PCA(n_components=50, random_state=42)
tfidf_reduced = pca.fit_transform(tfidf_features)

# K-Means on PCA-reduced data
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
kmeans_pca.fit(tfidf_reduced)
headlines['PCA_Cluster'] = kmeans_pca.labels_

print("\nComparison of Clusters with PCA-reduced data:")
for cluster in range(3):  # Analyze first three clusters
    print(f"\nCluster {cluster}:")
    print(headlines[headlines['PCA_Cluster'] == cluster].head(10)['headline'].values)

# Save results
headlines.to_csv("headlines_with_clusters.csv", index=False)