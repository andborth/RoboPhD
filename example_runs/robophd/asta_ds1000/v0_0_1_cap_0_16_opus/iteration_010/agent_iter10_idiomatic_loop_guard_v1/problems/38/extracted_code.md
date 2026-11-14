<code>
result = df[column_list].iloc[row_list].sum()
result = result.drop(result.idxmax())
</code>