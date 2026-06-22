<code>
result = df.loc[row_list, column_list].sum().astype(float).drop(df.loc[row_list, column_list].sum().idxmax())
</code>