<code>
mask = torch.arange(a.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)  # shape (10, 1000)
a = a.masked_fill(mask.unsqueeze(-1), 2333)
</code>