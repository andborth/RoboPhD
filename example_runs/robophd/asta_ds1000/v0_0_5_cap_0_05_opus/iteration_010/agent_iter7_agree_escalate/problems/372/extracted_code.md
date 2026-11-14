<code>
    from scipy.integrate import simpson
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.cos(X) ** 4 + np.sin(Y) ** 2
    result = simpson(simpson(Z, x=y, axis=1), x=x, axis=0)
    return result
</code>