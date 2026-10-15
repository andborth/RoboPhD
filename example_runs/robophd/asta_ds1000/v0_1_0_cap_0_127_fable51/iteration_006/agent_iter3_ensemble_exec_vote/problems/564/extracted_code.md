<code>
plt.plot(x, y)
plt.ylabel("Y")
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_left()
</code>