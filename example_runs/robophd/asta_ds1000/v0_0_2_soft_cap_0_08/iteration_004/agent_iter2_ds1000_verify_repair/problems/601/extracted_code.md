<code>
import matplotlib.pyplot as plt
current_ticks = plt.xticks()[0].tolist()
new_ticks = sorted(current_ticks + [2.1, 3, 7.6])
plt.xticks(new_ticks)
</code>