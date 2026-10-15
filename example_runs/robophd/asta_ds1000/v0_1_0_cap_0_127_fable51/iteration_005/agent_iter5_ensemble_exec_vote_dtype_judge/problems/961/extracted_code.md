<code>
idx = torch.arange(a.size(1)).unsqueeze(0)
mask = idx >= lengths.unsqueeze(1)
a[mask] = 2333
</code>