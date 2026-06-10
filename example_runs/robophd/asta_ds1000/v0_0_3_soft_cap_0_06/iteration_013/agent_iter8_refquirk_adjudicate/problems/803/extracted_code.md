<code>
dist = scipy.spatial.distance.cdist(centroids, data)
result = np.argmin(dist, axis=1)
</code>