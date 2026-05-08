<code>
# Concatenate df1 and df2
combined = pd.concat([df1, df2], axis=0, ignore_index=True)

# Fill city and district from df1 using id mapping
id_to_city = df1.set_index('id')['city']
id_to_district = df1.set_index('id')['district']

combined['city'] = combined['city'].fillna(combined['id'].map(id_to_city))
combined['district'] = combined['district'].fillna(combined['id'].map(id_to_district))

# Convert date to datetime and format
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%d-%b-%Y')

# Sort by id, then by date (ascending means earlier date comes after later date based on expected output)
# Looking at expected: id=3 has 01-Feb-2019 first, then 01-Jan-2019 -> descending date within id
combined['date_sort'] = pd.to_datetime(combined['date'], format='%d-%b-%Y')
combined = combined.sort_values(['id', 'date_sort'], ascending=[True, False]).drop(columns='date_sort')

# Select and reorder columns
result = combined[['id', 'city', 'district', 'date', 'value']].reset_index(drop=True)
</code>