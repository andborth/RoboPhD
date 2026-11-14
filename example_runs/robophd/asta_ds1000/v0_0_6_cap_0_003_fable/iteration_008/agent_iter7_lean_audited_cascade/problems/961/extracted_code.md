<code>
idx = torch.arange(a.size(1), device=a.device).view(1, -1, 1)
a = a.masked_fill(idx >= lengths.view(-1, 1, 1), 2333)
</code>