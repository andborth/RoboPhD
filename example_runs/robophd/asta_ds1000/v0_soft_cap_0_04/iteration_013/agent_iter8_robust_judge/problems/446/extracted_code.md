<code>
result = np.argsort(np.argsort(a, kind='stable')[::-1], kind='stable')
</code>