<code>
a[torch.arange(a.size(1))[None, :] < lengths[:, None]] = 0
</code>