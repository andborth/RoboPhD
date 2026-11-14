<code>
ax = plt.gca()
current_ticks = ax.get_xticks()
new_ticks = np.sort(np.concatenate([current_ticks, [2.1, 3, 7.6]]))
ax.set_xticks(new_ticks)
</code>