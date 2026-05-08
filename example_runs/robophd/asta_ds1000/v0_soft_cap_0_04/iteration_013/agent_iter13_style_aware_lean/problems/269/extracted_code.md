<code>
import numpy as np
df = pd.DataFrame(
    [df.values.ravel()],
    columns=pd.MultiIndex.from_product([df.index + 1, df.columns]).map('{0[1]}_{0[0]}'.format)
)
</code>