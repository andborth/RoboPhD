<code>
from scipy.spatial.distance import cdist

distances = cdist(data, centroids)
result = data[np.argmin(distances, axis=0)]
</code>