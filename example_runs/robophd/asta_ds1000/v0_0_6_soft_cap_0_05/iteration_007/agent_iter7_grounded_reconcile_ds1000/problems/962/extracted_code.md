<code>
a = a.masked_fill((torch.arange(a.size(1), device=a.device)[None, :] < lengths.to(a.device)[:, None]).unsqueeze(-1), 0)
</code>