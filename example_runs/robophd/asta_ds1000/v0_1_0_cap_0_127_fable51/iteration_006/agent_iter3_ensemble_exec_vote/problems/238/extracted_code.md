<code>
tmp = df2.merge(df1[['id', 'city', 'district']].drop_duplicates('id'), on='id', how='left')
result = pd.concat([tmp, df1], ignore_index=True, sort=False)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%d-%b-%Y')
result = result[['id', 'city', 'district', 'date', 'value']].sort_values('id', kind='mergesort').reset_index(drop=True)
</code>