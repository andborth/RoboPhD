<code>
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Convert square distance matrix to condensed distance matrix
condensed = squareform(data_matrix)

# Perform hierarchical clustering
Z = linkage(condensed, method='average')

# Cut the dendrogram to get 2 clusters
cluster_labels = list(fcluster(Z, 2, criterion='maxclust'))
</code>