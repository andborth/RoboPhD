<code>
dists = scipy.spatial.distance.cdist(data, centroids)
result = np.argmin(dists, axis=0)
</code>