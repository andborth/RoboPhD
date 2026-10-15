<code>
def smoothclamp(x, mi=x_min, mx=x_max):
    t = np.clip((x - mi) / (mx - mi), 0, 1)
    return mi + (mx - mi) * (t * t * (3 - 2 * t))
</code>