<code>
n = data.shape[1]
start = n % bin_size
bin_data_mean = data[:, start:].reshape(data.shape[0], -1, bin_size).mean(axis=2)
</code>