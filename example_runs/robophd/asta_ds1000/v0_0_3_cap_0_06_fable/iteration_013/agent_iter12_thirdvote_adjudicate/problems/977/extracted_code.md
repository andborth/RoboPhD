<code>
    y = torch.max(softmax_output, 1, keepdim=True)[1]
    return y
</code>