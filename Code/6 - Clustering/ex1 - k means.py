from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X, y = load_wine(return_X_y=True)

# scale data

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

sil_avg = []

# siloutte analysis for selecting optimal clusters

range_n_clusters = range(2, 10)

# compute the average siloutte score for each cluster number to determine optimal number of clusters

for n in range_n_clusters:
    # create kmeans clustering object
    clusterer = KMeans(n_clusters=n, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    siloutte_avg = silhouette_score(X, cluster_labels)

    sil_avg.append(siloutte_avg)

plt.plot(range_n_clusters, sil_avg)
plt.show()

# k-means with optimal clusters from above

k_optimal = range_n_clusters[sil_avg.index(max(sil_avg))]

clusterer = KMeans(n_clusters=k_optimal, random_state=10)

clusterer.fit(X)

# show some of the clustering metrics for validity
print(metrics.homogeneity_score(y, clusterer.predict(X)))
print(metrics.completeness_score(y, clusterer.predict(X)))
print(metrics.v_measure_score(y, clusterer.predict(X)))
print(metrics.adjusted_rand_score(y, clusterer.predict(X)))
print(metrics.adjusted_mutual_info_score(y, clusterer.predict(X)))
print(metrics.silhouette_score(X, clusterer.predict(X)))




