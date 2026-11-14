<code>
distances = scipy.spatial.distance.cdist(centroids, data)
closest_indices = np.argmin(distances, axis=1)
result = data[closest_indices]
</code>