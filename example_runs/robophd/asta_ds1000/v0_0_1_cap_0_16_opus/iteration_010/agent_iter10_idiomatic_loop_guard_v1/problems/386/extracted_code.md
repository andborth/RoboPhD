<code>
from numpy.lib.stride_tricks import sliding_window_view
patch = 2
result = sliding_window_view(a, (patch, patch)).reshape(-1, patch, patch)
</code>