<code>
def smoothclamp(x, x_min, x_max):
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max")

    t = (x - x_min) / (x_max - x_min)
    t = np.clip(t, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # 3t^2 - 2t^3
    return x_min + s * (x_max - x_min)

result = smoothclamp(x, x_min, x_max)
print(result)
</code>