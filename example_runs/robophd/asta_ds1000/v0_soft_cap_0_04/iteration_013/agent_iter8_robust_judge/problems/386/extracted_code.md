<code>
from numpy.lib.stride_tricks import sliding_window_view

patches = sliding_window_view(a, (2, 2))
result = patches.reshape(-1, 2, 2)
</code>