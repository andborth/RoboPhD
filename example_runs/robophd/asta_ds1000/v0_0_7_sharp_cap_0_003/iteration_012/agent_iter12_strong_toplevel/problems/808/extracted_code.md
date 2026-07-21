<code>
norm_const = integrate.quad(lambda t: bekkers(t, estimated_a, estimated_m, estimated_d), range_start, range_end)[0]

def bekkers_cdf(x):
    x = np.asarray(x, dtype=float)
    cdf = np.empty_like(x, dtype=float)
    cdf[x <= range_start] = 0.0
    cdf[x >= range_end] = 1.0
    mask = (x > range_start) & (x < range_end)
    cdf[mask] = [integrate.quad(lambda t: bekkers(t, estimated_a, estimated_m, estimated_d), range_start, xi)[0] / norm_const for xi in x[mask]]
    return cdf if cdf.ndim else float(cdf)

result = stats.kstest(sample_data, bekkers_cdf)
</code>