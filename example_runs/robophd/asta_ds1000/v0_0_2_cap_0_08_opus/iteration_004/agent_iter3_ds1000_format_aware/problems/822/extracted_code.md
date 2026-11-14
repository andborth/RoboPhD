<code>
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded = pd.DataFrame(mlb.fit_transform(df['Col3']), columns=mlb.classes_, index=df.index)
df_out = df.drop(columns=['Col3']).join(encoded)
</code>