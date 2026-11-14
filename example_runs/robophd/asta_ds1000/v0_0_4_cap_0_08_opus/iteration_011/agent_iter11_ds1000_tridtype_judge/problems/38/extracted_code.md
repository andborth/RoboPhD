<code>
res = df.loc[row_list, column_list].sum().astype(float)
result = res.drop(res.idxmax())
</code>