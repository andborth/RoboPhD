<code>
d = scipy.spatial.distance.cdist(centroids, data)
result = data[np.argmin(d, axis=1)]
</code>