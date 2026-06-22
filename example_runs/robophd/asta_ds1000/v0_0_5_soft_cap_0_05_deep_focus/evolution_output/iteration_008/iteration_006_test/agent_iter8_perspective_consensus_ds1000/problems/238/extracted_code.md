<code>
result = pd.concat(
    [
        df1,
        df2.merge(df1[['id', 'city', 'district']], on='id', how='left')
    ],
    ignore_index=True
)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%d-%b-%Y')
result = result.sort_values(['id', 'date'], kind='mergesort')[['id', 'city', 'district', 'date', 'value']].reset_index(drop=True)
</code>