<code>
from sklearn.preprocessing import LabelEncoder

new_X = pd.DataFrame(X).apply(LabelEncoder().fit_transform).to_numpy()
</code>