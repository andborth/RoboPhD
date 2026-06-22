<code>
result = df[pd.to_numeric(df["A"], errors="coerce").notna()]
</code>