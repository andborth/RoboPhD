<code>
df_temp = df.copy()
df_temp.index = range(1, len(df_temp) + 1)
s = df_temp.stack()
s.index = [f'{col}_{row}' for row, col in s.index]
df = s.to_frame().T.reset_index(drop=True)
</code>