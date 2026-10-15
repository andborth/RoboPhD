<code>
result = df[column_list].iloc[row_list].sum(axis=0)
result = result.drop(result.idxmax())
</code>