<code>
df = df.stack().to_frame().T
df.columns = [f'{col}_{i+1}' for i in range(len(df.index.levels[0])) for col in df.columns.get_level_values(0)[:len(df.columns)//len(df.index.levels[0])]]
</code>