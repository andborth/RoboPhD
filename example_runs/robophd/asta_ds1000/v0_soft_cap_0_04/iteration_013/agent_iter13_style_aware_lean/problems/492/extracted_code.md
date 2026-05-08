<code>
result = np.sum(~(np.isclose(s1, s2) | (np.isnan(s1) & np.isnan(s2))))
</code>