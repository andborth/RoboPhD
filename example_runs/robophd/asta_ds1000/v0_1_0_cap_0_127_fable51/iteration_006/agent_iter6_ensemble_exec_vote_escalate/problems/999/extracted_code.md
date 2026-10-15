<code>
W = W.unsqueeze(0).unsqueeze(2)
result = torch.bmm(data, W.expand(data.size(0), W.size(1), 1)).squeeze(-1)
result = result.view(10, 2, 3)
</code>