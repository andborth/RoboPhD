<code>
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

cluster_labels = scipy.cluster.hierarchy.fcluster(
    scipy.cluster.hierarchy.linkage(squareform(data_matrix, checks=False), method='ward'),
    2,
    criterion='maxclust'
).tolist()
</code>