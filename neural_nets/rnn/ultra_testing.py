import tensorflow as tf

def elements_gen():
    sequence = [
        [[11, 22], [22, 22], [33, 22]],
        [[33, 22], [44, 22], [55, 22], [66, 22], [77, 22]],
        [[11, 22], [22, 22]],
        [[88, 22], [99, 22], [11, 22], [22, 22]],
    ]

    label = [1, 2, 1, 2]
    for x, y in zip(sequence, label):
        yield (x, y)


def element_length_fn(x, y):
    return tf.shape(x)[0]


dataset = tf.data.Dataset.from_generator(
    generator=elements_gen, output_shapes=([None, 2], []), output_types=(tf.int32, tf.int32)
)

dataset = dataset.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=element_length_fn,
        bucket_batch_sizes=[2, 2, 2],
        bucket_boundaries=[0, 5],
        padding_values=(0,0),
    )
)

batch = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    for _ in range(2):
        print("Get_next:")
        print(sess.run(batch))