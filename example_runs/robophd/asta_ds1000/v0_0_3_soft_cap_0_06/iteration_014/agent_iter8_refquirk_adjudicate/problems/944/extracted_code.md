<code>
C = torch.index_select(B, 1, (A_log == 0).nonzero().view(-1))
</code>