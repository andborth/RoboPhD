<code>
    if isinstance(softmax_output, torch.Tensor):
        result = torch.argmax(softmax_output, dim=1, keepdim=True)
    else:
        result = np.argmax(softmax_output, axis=1).reshape(-1, 1)
    return result
</code>