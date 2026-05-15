<code>
distances = scipy.spatial.distance.cdist(centroids, data)
result = np.argmin(distances, axis=1)
</code>