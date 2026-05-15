<code>
df.index += 1
df = df.stack()
df.index = [f"{col}_{idx}" for idx, col in df.index]
df = df.to_frame().T.reset_index(drop=True)
</code>