<code>
interp = scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], z)
result = interp(s, t, grid=False)
</code>