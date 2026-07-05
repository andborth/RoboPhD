<code>
import numpy as np

mean = np.mean(a)
std = np.std(a, ddof=0)
fourth_moment = np.mean((a - mean)**4)
kurtosis_result = fourth_moment / (std**4)
</code>