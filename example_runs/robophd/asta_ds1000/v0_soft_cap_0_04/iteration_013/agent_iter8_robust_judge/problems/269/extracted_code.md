<code>
df = pd.DataFrame([df.values.flatten()], columns=[f"{col}_{i+1}" for i in range(len(df)) for col in df.columns])
</code>