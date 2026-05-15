<code>
centroids_indices = []
for centroid in centroids:
    distances = scipy.spatial.distance.cdist([centroid], data)
    centroids_indices.append(np.argmin(distances))
result = centroids_indices
</code>