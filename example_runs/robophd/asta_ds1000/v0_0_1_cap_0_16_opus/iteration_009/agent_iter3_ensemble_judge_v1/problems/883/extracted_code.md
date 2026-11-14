<code>
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Convert square distance matrix to condensed form
condensed = squareform(data_matrix)

# Perform hierarchical clustering using average linkage
Z = linkage(condensed, method='average')

# Get cluster labels for 2 clusters
cluster_labels = fcluster(Z, 2, criterion='maxclust').tolist()
</code>