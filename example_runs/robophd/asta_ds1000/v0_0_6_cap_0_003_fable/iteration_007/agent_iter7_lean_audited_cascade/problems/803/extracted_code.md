<code>
from scipy.spatial.distance import cdist

# Calculate distances from each data point to each centroid
distances = cdist(data, centroids)

# For each centroid, find the index of the closest data point
result = np.argmin(distances, axis=0)
</code>