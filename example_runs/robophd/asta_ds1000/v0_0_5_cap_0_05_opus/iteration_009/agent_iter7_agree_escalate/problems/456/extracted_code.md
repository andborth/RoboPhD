<code>
result = np.linalg.norm(a[:, np.newaxis, :] - a[np.newaxis, :, :], axis=2)
</code>