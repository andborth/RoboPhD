<code>
s = df.loc[row_list, column_list].sum()
result = s.drop(s.idxmax())
</code>