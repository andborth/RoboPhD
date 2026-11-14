<code>
result = (
    df.groupby('name', sort=False)
      .apply(lambda g1: g1.groupby('v1', sort=False)
            .apply(lambda g2: dict(zip(g2['v2'], g2['v3'])))
            .to_dict())
      .to_dict()
)
</code>