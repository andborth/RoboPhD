<code>
n_bins = data.shape[1] // bin_size
trimmed = data[:, data.shape[1] - n_bins * bin_size:]
bin_data_mean = trimmed.reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
</code>