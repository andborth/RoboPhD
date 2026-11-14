<code>
mask = torch.arange(a.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
a[mask] = 2333
</code>