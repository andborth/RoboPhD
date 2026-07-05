<code>
mask = torch.arange(a.size(1), device=a.device).unsqueeze(0) >= lengths.unsqueeze(1)
a[mask.unsqueeze(-1).expand_as(a)] = 2333
</code>