<code>
def to_nested(df):
    if df.shape[1] == 1:
        if df.shape[0] == 1:
            return df.iloc[0, 0]
        return df.iloc[:, 0].tolist()
    return {k: to_nested(g.iloc[:, 1:]) for k, g in df.groupby(df.columns[0])}

result = to_nested(df)
</code>