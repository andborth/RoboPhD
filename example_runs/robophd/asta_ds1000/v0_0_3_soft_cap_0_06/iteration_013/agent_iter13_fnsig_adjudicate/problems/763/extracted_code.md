<code>
f = scipy.interpolate.RectBivariateSpline(x.ravel(), y.ravel(), z)
result = f(s, t, grid=False)
</code>