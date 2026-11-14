<code>
def smoothclamp(x, mi, mx):
    t = np.clip((x - mi) / (mx - mi), 0, 1)
    return mi + (mx - mi) * (3 * t**2 - 2 * t**3)
</code>