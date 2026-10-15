<code>
spline = scipy.interpolate.RectBivariateSpline(x.ravel(), y.ravel(), z)
result = spline.ev(s, t)
</code>