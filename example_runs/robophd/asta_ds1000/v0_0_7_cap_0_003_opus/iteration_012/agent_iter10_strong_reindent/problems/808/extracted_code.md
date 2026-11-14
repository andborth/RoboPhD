<code>
norm_const = integrate.quad(lambda t: bekkers(t, estimated_a, estimated_m, estimated_d), range_start, range_end)[0]

def bekkers_cdf(x):
    x = np.asarray(x, dtype=float)
    def cdf_scalar(xi):
        if xi <= range_start:
            return 0.0
        if xi >= range_end:
            return 1.0
        return integrate.quad(lambda t: bekkers(t, estimated_a, estimated_m, estimated_d), range_start, xi)[0] / norm_const
    return np.vectorize(cdf_scalar, otypes=[float])(x)

result = stats.kstest(sample_data, bekkers_cdf)
</code>