<code>
from numbers import Integral

df = [v for v in df["Field1"] if isinstance(v, Integral) and not isinstance(v, bool)]
</code>