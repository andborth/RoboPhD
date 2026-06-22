<code>
import numbers
df = df.loc[df["Field1"].apply(lambda x: isinstance(x, numbers.Integral)), "Field1"].tolist()
</code>