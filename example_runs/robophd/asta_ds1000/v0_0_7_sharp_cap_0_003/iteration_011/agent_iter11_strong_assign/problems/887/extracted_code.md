<code>
from sklearn.preprocessing import StandardScaler, PowerTransformer

centered_scaled_data = PowerTransformer(standardize=True).fit_transform(data)
</code>