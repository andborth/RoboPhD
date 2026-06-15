<code>
def smoothclamp(x, min=x_min, max=x_max):
    t = np.clip((np.asarray(x) - min) / (max - min), 0, 1)
    return min + (max - min) * (3 * t**2 - 2 * t**3)
</code>