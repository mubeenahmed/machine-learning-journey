# Source code taken from https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


# Generate sample data
np.random.seed(0)
X, labels_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

# When visualize, it is easy to pick the cluster because sklearn 
# do it automatically
plt.scatter(X[:, 0], X[:, 1], s=50);


# Visualizing the cluster by coloring
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Color each cluster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
