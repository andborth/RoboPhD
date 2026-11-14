<code>
indices = torch.arange(1000).unsqueeze(0).unsqueeze(2)  # (1, 1000, 1)
lengths_expanded = lengths.unsqueeze(1).unsqueeze(2)    # (10, 1, 1)
mask = indices < lengths_expanded                        # (10, 1000, 1)
a = a.masked_fill(mask, 0)
</code>