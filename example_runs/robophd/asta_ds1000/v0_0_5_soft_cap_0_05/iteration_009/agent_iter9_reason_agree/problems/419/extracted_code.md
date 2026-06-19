<code>
n_rows = data.shape[0]
trim = data.shape[1] % bin_size
aligned = data[:, trim:]
n_bins = aligned.shape[1] // bin_size
bin_data_mean = aligned.reshape(n_rows, n_bins, bin_size).mean(axis=2)
</code>