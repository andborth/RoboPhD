<code>
for r in products:
    df.loc[df['product'].between(r[0], r[1]), 'score'] *= 10
</code>