<code>
mask = torch.arange(a.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
a = a.masked_fill(mask.unsqueeze(-1), 2333)
</code>