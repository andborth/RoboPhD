<code>
mask = torch.arange(a.size(1), device=a.device)[None, :] < lengths[:, None]
a[mask] = 0
</code>