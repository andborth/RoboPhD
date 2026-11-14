<code>
s = df.stack()
s.index = s.index.map(lambda t: str(t[1]) + '_' + str(t[0] + 1))
df = s.to_frame().T
</code>