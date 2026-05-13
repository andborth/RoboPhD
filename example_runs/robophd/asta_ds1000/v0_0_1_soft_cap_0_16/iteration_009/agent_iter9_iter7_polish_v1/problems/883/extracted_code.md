<code>
Z = scipy.cluster.hierarchy.linkage(np.array(data_matrix), method='ward')
cluster_labels = scipy.cluster.hierarchy.fcluster(Z, t=2, criterion='maxclust').tolist()
cluster_labels = [x - 1 for x in cluster_labels]
</code>