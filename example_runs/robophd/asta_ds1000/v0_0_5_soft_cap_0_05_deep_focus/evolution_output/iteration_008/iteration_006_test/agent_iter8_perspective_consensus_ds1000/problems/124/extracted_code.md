<code>
df["keywords_all"] = df.apply(lambda row: "-".join(row.dropna().astype(str)), axis=1)
</code>