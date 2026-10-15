<code>
def g(a, b):
    return tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)

result = g(a, b)
</code>