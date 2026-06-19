<code>
from scipy.signal import lfilter

B = pd.Series(lfilter([a], [1, -b], A.to_numpy()), index=A.index, name=A.name)
</code>