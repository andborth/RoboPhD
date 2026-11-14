<code>
dist_mat = np.asarray(data_matrix, dtype=float)
condensed = scipy.spatial.distance.squareform(dist_mat)
Z = scipy.cluster.hierarchy.linkage(condensed, method='average')
cluster_labels = scipy.cluster.hierarchy.fcluster(Z, 2, criterion='maxclust').tolist()
</code>