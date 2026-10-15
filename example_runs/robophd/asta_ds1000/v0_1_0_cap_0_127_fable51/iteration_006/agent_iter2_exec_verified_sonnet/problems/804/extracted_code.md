<code>
dist = scipy.spatial.distance.cdist(centroids, data)
closest_indices = dist.argmin(axis=1)
result = data[closest_indices]
</code>