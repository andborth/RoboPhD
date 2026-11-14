<code>
        return torch.argmax(softmax_output, dim=1, keepdim=True)

    y = solve(softmax_output)
</code>