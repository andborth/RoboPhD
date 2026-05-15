<code>
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded = pd.DataFrame(mlb.fit_transform(df['Col3']), columns=mlb.classes_, index=df.index)
df_out = pd.concat([df.drop('Col3', axis=1), encoded], axis=1)
</code>