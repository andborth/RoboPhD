<code>
mask = torch.arange(1000).unsqueeze(0) < lengths.unsqueeze(1)
a[mask] = 0
</code>