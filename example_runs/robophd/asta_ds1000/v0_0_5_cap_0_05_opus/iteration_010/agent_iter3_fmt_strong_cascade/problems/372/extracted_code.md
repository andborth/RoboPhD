<code>
    from scipy import integrate
    X, Y = np.meshgrid(x, y, indexing='ij')
    z = np.cos(X) ** 4 + np.sin(Y) ** 2
    return integrate.simpson(integrate.simpson(z, x=y, axis=1), x=x, axis=0)
</code>