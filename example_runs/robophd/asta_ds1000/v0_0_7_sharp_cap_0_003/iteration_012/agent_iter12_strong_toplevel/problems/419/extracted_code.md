<code>
n = data.shape[1] // bin_size * bin_size
bin_data_mean = data[:, -n:].reshape(data.shape[0], -1, bin_size).mean(axis=2)
</code>