<code>
stacked = df.stack()
stacked.index = [f'{col}_{row+1}' for row, col in stacked.index]
df = pd.DataFrame(stacked).T.reset_index(drop=True)
</code>