<code>
df["keywords_all"] = df.apply(lambda x: "-".join(x.dropna()), axis=1)
</code>