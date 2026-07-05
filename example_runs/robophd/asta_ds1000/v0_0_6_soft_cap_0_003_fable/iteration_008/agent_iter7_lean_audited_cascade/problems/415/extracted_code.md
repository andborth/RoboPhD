<code>
bin_data_max = data[:len(data) - len(data) % bin_size].reshape(-1, bin_size).max(axis=1)
</code>