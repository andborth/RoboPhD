<code>
new_shape = data.shape[1] // bin_size
bin_data_mean = data[:, -new_shape*bin_size:].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
</code>