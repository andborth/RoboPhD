<code>
    df = df.copy()
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    return df
</code>