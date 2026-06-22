<code>
B = pd.Series(np.asarray(__import__('scipy.signal').signal.lfilter([a], [1, -b], A.to_numpy())), index=A.index)
</code>