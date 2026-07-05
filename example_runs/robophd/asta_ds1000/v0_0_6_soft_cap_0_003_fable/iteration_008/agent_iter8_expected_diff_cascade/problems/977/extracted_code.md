<code>
    if isinstance(softmax_output, torch.Tensor):
        result = torch.argmax(softmax_output, dim=1, keepdim=True)
    else:
        arr = np.asarray(softmax_output)
        result = np.argmax(arr, axis=1).reshape(-1, 1)
    return result
</code>