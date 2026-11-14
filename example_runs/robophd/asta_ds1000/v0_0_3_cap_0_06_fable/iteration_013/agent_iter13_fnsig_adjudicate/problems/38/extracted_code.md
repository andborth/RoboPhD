<code>
s = df.loc[row_list, column_list].sum().astype(float)
result = s.drop(s.idxmax())
</code>