<code>
df = pd.DataFrame([df.to_numpy().ravel()],
                  columns=[f"{col}_{i}" for i in range(1, len(df) + 1) for col in df.columns])
</code>