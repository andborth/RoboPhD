<code>
from numpy.lib.stride_tricks import sliding_window_view
result = sliding_window_view(a, (2, 2)).reshape(-1, 2, 2)
</code>