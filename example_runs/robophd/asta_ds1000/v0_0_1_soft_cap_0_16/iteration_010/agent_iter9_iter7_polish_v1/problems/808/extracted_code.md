<code>
def bekkers_cdf(x, a, m, d, range_start, range_end):
    values = []
    for value in x:
        integral = integrate.quad(bekkers, range_start, value, args=(a, m, d))[0]
        normalized = integral / integrate.quad(bekkers, range_start, range_end, args=(a, m, d))[0]
        values.append(normalized)
    return np.array(values)

result = stats.kstest(sample_data, lambda x: bekkers_cdf(x, estimated_a, estimated_m, estimated_d, range_start, range_end))
</code>