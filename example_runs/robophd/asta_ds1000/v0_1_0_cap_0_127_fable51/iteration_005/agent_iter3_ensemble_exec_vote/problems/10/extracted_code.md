<code>
def nested_dict(frame):
    if frame.shape[1] == 2:
        return frame.set_index(frame.columns[0])[frame.columns[1]].to_dict()
    return frame.groupby(frame.columns[0], sort=False).apply(lambda x: nested_dict(x.iloc[:, 1:])).to_dict()

result = nested_dict(df)
</code>