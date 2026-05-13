<code>
n_rows, n_cols = data.shape
n_bins = n_cols // bin_size
trim = n_cols % bin_size
trimmed = data[:, trim:]  # aligned to end: discard first `trim` elements
reshaped = trimmed.reshape(n_rows, n_bins, bin_size)
bin_data_mean = reshaped.mean(axis=2)
</code>