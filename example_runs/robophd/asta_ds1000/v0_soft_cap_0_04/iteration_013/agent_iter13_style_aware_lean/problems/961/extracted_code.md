<code>
mask = torch.arange(a.size(1), device=a.device).unsqueeze(0) >= lengths.to(a.device).unsqueeze(1)
a = a.masked_fill(mask.unsqueeze(-1), 2333)
</code>