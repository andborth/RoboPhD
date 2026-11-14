<code>
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_out = df.join(pd.DataFrame(mlb.fit_transform(df['Col3']), columns=mlb.classes_, index=df.index)).drop(columns='Col3')
</code>