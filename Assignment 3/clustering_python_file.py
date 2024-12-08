  
 ### importing libraries

 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

 
data = np.load('tfidf_features.npy',allow_pickle=True)
print(data.shape)
data
 
headlines_df = pd.read_csv('headlines.csv').set_index("idx")
headlines_df = headlines_df['headline'].astype(str)
headlines_df.info()

### K-Means Clustering
 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=50)
data_pca = pca.fit_transform(data_scaled)
print(data_pca.shape)
print(f"variance : {sum(pca.explained_variance_ratio_) * 100}")

start = 2
end = 200+1
km = KMeans(random_state=42,init='k-means++',max_iter=700)
visualizer = KElbowVisualizer(km, k=(start,end)) #visualizes elbow
visualizer.fit(data_pca)
visualizer.show()
 
km_1 = KMeans(random_state=42,init='k-means++',max_iter=700,n_clusters=visualizer.elbow_value_)
km_1.fit(data_pca)
print(f"labels : {len(km_1.labels_)}\noptimal k : {visualizer.elbow_value_}\nscore : {visualizer.elbow_score_}",end="\n\n")
Km_1_headlines_df = pd.DataFrame({
    'headline' : headlines_df,
    'cluster' : km_1.labels_
})
for cid,hl_grp in Km_1_headlines_df.groupby('cluster'):
    print(f"cluster Id : {cid}")
    print('\n'.join(hl_grp.head(10)['headline'].tolist()),end='\n\n')
 
np.random.seed(43)
rand_ind = np.random.choice(data_pca.shape[0], size=1000, replace=False)
linkage_matrix = linkage(data_pca[rand_ind], method='ward')
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix, labels=np.arange(len(data_pca[rand_ind])), leaf_rotation=90, leaf_font_size=0,truncate_mode='lastp', p=500,color_threshold=40)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show() 
 
silhouette_scores = []
calinski_scores = []
davies_bouldin_scores = []
for i in range(start,end):
    fc = fcluster(linkage_matrix, i, criterion='maxclust')
    silhouette_scores.append(silhouette_score(data_pca[rand_ind], fc))
    calinski_scores.append(calinski_harabasz_score(data_pca[rand_ind], fc))
    davies_bouldin_scores.append(davies_bouldin_score(data_pca[rand_ind], fc))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

x_axis_plt = range(start,end)
ax1.plot(x_axis_plt, silhouette_scores, marker='o')
ax1.set_title('Silhouette Score')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Silhouette Score')

ax2.plot(x_axis_plt, calinski_scores, marker='o')
ax2.set_title('Calinski-Harabasz Index')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Calinski-Harabasz Score')

ax3.plot(x_axis_plt, davies_bouldin_scores, marker='o')
ax3.set_title('Davies-Bouldin Index')
ax3.set_xlabel('Number of Clusters')
ax3.set_ylabel('Davies-Bouldin Score')

plt.tight_layout()
plt.show()

silhouette_optimal_k = np.argmax(silhouette_scores) + start

calinski_optimal_k = np.argmax(calinski_scores) + start

davies_bouldin_optimal_k = np.argmin(davies_bouldin_scores) + start

print(f"optimal K : \n silhouette : {silhouette_optimal_k}\n calinski : {calinski_optimal_k}\n davies : {davies_bouldin_optimal_k}")

opt_k = calinski_optimal_k
clustering = AgglomerativeClustering(n_clusters=opt_k, linkage='ward',compute_full_tree=True)
labels = clustering.fit_predict(data_pca[rand_ind])
print(len(labels))
 
hc_headlines_df = pd.DataFrame({
    'headline' : headlines_df[rand_ind],
    'cluster' : labels
})

for cid,hl_grp in hc_headlines_df.groupby('cluster'):
    print(f"cluster Id : {cid}")
    print('\n'.join(hl_grp.head(10)['headline'].tolist()),end='\n\n')