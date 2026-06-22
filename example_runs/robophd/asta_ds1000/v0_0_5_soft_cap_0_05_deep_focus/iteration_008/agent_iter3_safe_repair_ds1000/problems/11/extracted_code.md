<code>
df = df.assign(datetime=df['datetime'].dt.tz_localize(None))
</code>