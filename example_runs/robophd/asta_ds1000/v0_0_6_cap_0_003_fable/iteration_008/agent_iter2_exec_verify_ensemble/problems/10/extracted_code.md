<code>
def df_to_nested_dict(df):
    result = {}
    key_cols = df.columns[:-1]
    value_col = df.columns[-1]

    for row in df.itertuples(index=False, name=None):
        d = result
        for key in row[:-2]:
            d = d.setdefault(key, {})
        d.setdefault(row[-2], {})
        d[row[-2]] = row[-1] if not isinstance(d[row[-2]], dict) else d[row[-2]]
        if isinstance(d[row[-2]], dict):
            d[row[-2]] = row[-1]

    return result

result = {}
for row in df.itertuples(index=False, name=None):
    d = result
    for key in row[:-2]:
        d = d.setdefault(key, {})
    d[row[-2]] = row[-1]
</code>