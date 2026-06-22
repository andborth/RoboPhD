<code>
interp = scipy.interpolate.RegularGridInterpolator((x[:, 0], y[0]), z)
result = interp(np.column_stack((s, t)))
</code>