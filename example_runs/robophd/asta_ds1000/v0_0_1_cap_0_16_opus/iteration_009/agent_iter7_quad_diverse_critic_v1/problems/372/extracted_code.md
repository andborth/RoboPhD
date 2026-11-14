<code>
    from scipy.integrate import simpson
    z = np.cos(x[:,None])**4 + np.sin(y[None,:])**2
    result = simpson(simpson(z, x=y, axis=1), x=x)
    return result
</code>