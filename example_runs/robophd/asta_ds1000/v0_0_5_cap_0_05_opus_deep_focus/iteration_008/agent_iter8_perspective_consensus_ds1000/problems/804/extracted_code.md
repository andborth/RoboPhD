<code>
dist = scipy.spatial.distance.cdist(centroids, data)
result = data[np.argmin(dist, axis=1)]
</code>