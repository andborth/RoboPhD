<code>
tmp = df2.merge(df1[['id', 'city', 'district']], on='id', how='left')
result = pd.concat([df1, tmp[df1.columns]], ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%d-%b-%Y')
result = result.sort_values(['id', 'date'], ignore_index=True)
</code>