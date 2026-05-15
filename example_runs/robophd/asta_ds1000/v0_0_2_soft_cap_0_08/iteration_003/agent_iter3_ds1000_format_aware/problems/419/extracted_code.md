<code>
n_cols = data.shape[1]
trim = n_cols % bin_size
trimmed = data[:, trim:] if trim > 0 else data
n_bins = trimmed.shape[1] // bin_size
bin_data_mean = trimmed.reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
</code>