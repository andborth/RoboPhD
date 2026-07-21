<code>
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

Z = linkage(squareform(np.array(data_matrix)), method='average')
cluster_labels = fcluster(Z, 2, criterion='maxclust').tolist()
</code>