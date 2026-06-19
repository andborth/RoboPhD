<code>
    try:
        from scipy.integrate import simpson as _simpson
    except ImportError:
        from scipy.integrate import simps as _simpson
    z = np.cos(x)[None, :]**4 + np.sin(y)[:, None]**2
    result = _simpson(_simpson(z, x=x, axis=1), x=y, axis=0)
    return result
</code>