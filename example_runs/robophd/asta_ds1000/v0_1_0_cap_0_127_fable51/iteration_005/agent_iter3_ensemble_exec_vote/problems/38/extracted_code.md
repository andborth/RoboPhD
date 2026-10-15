<code>
result = df.loc[row_list, column_list].sum()
result = result.drop(result.idxmax())
</code>