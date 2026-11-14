<code>
for lo, hi in products:
    df.loc[(df['product'] >= lo) & (df['product'] <= hi), 'score'] *= 10
</code>