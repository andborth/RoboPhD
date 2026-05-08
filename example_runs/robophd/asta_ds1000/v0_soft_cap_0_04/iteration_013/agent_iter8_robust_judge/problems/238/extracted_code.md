<code>
# Concatenate df1 and df2
combined = pd.concat([df1, df2], axis=0, ignore_index=True)

# Fill city and district from df1 using id mapping
id_to_city = df1.set_index('id')['city']
id_to_district = df1.set_index('id')['district']

combined['city'] = combined['city'].fillna(combined['id'].map(id_to_city))
combined['district'] = combined['district'].fillna(combined['id'].map(id_to_district))

# Convert date to datetime and format
combined['date'] = pd.to_datetime(combined['date'])
combined['date'] = combined['date'].dt.strftime('%d-%b-%Y')

# Sort by id, then by date descending (newer date first within same id)
combined = combined.sort_values(['id', 'date'], ascending=[True, False]).reset_index(drop=True)

# Reorder columns
result = combined[['id', 'city', 'district', 'date', 'value']]
</code>