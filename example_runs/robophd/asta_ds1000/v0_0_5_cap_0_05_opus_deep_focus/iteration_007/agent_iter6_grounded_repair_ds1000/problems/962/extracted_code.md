<code>
mask = torch.arange(a.size(1), device=a.device).unsqueeze(0) < lengths.to(a.device).unsqueeze(1)
a[mask] = 0
</code>