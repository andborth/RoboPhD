<code>
mask = torch.arange(a.size(1), device=a.device)[None, :] >= lengths[:, None]
a = a.masked_fill(mask[:, :, None], 2333)
</code>