<code>
distances = scipy.spatial.distance.cdist(data, centroids)
labels = np.argmin(distances, axis=1)
result = np.array([
    data[labels == i][np.argmin(distances[labels == i, i])]
    for i in range(len(centroids))
])
</code>