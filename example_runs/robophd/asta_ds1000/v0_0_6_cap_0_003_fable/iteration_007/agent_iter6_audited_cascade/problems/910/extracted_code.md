<code>
    le = LabelEncoder()
    df = df.copy()
    df['Sex'] = le.fit_transform(df['Sex'])
    return df
</code>