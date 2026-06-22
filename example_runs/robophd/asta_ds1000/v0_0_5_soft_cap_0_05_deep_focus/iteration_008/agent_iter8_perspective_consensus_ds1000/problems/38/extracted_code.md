<code>
result = df.loc[row_list, column_list].sum().drop(df.loc[row_list, column_list].sum().idxmax())
</code>