<code>
n_bins = data.shape[1] // bin_size
start = data.shape[1] - n_bins * bin_size

trimmed = data[:, start:]
bin_data_mean = trimmed.reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
</code>