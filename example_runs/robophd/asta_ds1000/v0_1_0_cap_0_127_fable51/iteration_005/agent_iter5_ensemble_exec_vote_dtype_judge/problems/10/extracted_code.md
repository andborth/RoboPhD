<code>
def nested(df):
    if df.shape[1] == 2:
        return df.set_index(df.columns[0])[df.columns[1]].to_dict()
    return df.groupby(df.columns[0], sort=False)[df.columns[1:]].apply(nested).to_dict()

result = nested(df)
</code>