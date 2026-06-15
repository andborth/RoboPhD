<code>
def smoothclamp(x, x_min=x_min, x_max=x_max):
    t = np.clip((np.asarray(x) - x_min) / (x_max - x_min), 0, 1)
    y = x_min + (x_max - x_min) * (3 * t**2 - 2 * t**3)
    return y.item() if np.ndim(y) == 0 else y
</code>