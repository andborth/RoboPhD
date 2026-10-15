<code>
interp = scipy.interpolate.RectBivariateSpline(x.ravel(), y.ravel(), z)
result = interp(s, t, grid=False)
</code>