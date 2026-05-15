<code>
mask = torch.arange(a.shape[1]).view(1, -1, 1) < lengths.view(-1, 1, 1)
a = a.masked_fill(mask, 0)
</code>