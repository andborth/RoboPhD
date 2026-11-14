<code>
trim = data.shape[1] % bin_size
aligned = data[:, trim:] if trim else data
bin_data_mean = aligned.reshape(data.shape[0], -1, bin_size).mean(axis=2)
</code>