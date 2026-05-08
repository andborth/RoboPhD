<code>
# Concatenate df1 and df2
combined = pd.concat([df1, df2], axis=0, sort=False)

# Fill city and district from df1 using id mapping
id_to_city = df1.set_index('id')['city']
id_to_district = df1.set_index('id')['district']

combined['city'] = combined['city'].fillna(combined['id'].map(id_to_city))
combined['district'] = combined['district'].fillna(combined['id'].map(id_to_district))

# Convert date to datetime
combined['date'] = pd.to_datetime(combined['date'])

# Sort by id, then by date descending (larger date first within same id)
combined = combined.sort_values(['id', 'date'], ascending=[True, False]).reset_index(drop=True)

# Format date
combined['date'] = combined['date'].dt.strftime('%d-%b-%Y')

# Reorder columns
result = combined[['id', 'city', 'district', 'date', 'value']].reset_index(drop=True)
</code>