<code>
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded = mlb.fit_transform(df['Col3'])
df_out = df[['Col1', 'Col2']].join(
    pd.DataFrame(encoded, columns=mlb.classes_, index=df.index)
)
</code>