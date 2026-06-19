<code>
condensed = scipy.cluster.hierarchy.distance.squareform(np.array(data_matrix))
Z = scipy.cluster.hierarchy.linkage(condensed, method='single')
cluster_labels = scipy.cluster.hierarchy.fcluster(Z, 2, criterion='maxclust').tolist()
</code>