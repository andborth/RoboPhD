<code>
def df_to_dict(df):
    cols = df.columns.tolist()
    if len(cols) == 2:
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return df.groupby(cols[0]).apply(
        lambda x: df_to_dict(x[cols[1:]]),
        include_groups=False
    ).to_dict()

result = df_to_dict(df)
</code>