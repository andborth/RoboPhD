<code>
for low, high in products:
    df.loc[(df['product'] >= low) & (df['product'] <= high), 'score'] *= 10
</code>