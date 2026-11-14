<code>
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

cluster_labels = fcluster(linkage(squareform(np.array(data_matrix)), method="average"), 2, criterion="maxclust").tolist()
</code>