<code>
df = df.stack().to_frame().T
df.columns = [f'{col}_{idx+1}' for idx, col in df.columns]
df = df.reset_index(drop=True)
</code>