<code>
n_cols = data.shape[1]
remainder = n_cols % bin_size
trimmed = data[:, remainder:]
n_bins = trimmed.shape[1] // bin_size
reshaped = trimmed.reshape(data.shape[0], n_bins, bin_size)
bin_data_mean = reshaped.mean(axis=2)
</code>