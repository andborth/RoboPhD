<code>
trimmed = data[:(len(data) // bin_size) * bin_size]
bin_data_max = trimmed.reshape(-1, bin_size).max(axis=1)
</code>