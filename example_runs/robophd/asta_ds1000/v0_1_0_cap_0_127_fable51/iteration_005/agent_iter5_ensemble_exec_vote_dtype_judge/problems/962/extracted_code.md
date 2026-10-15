<code>
lengths = torch.as_tensor(lengths, device=a.device)
mask = torch.arange(a.size(1), device=a.device).unsqueeze(0) < lengths.unsqueeze(1)
a[mask] = 0
</code>