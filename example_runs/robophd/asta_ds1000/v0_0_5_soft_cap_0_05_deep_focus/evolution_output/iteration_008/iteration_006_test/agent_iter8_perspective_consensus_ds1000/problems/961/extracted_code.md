<code>
a[torch.arange(a.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)] = 2333
</code>