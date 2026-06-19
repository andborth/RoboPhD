<code>
dist = scipy.spatial.distance.cdist(centroids, data)
result = data[dist.argmin(axis=1)]
</code>