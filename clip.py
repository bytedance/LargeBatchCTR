import tensorflow as tf


def clip_id_norm(w, g, ratio=1, ids=None, cnts=None, min_w=0.03, const=False):
    if isinstance(g, tf.IndexedSlices):
        # FIXME: This part is not tested
        values = tf.convert_to_tensor(g.values)
        clipnorm = tf.norm(tf.gather(w, g.indices), axis=-1)
    else:
        values = g
        if const:
            clipnorm = tf.constant([min_w] * g.shape[0])
        else:
            clipnorm = tf.norm(w, axis=-1)
            # bound weight norm by min_w
            clipnorm = tf.maximum(clipnorm, min_w)
        # scale by cnting
        cnts = tf.tensor_scatter_nd_update(
            tf.ones([clipnorm.shape[0]], dtype=tf.int32),
            tf.expand_dims(ids, -1),
            cnts,
        )
        clipnorm = clipnorm * tf.cast(cnts, tf.float32)

    clip_t = ratio * clipnorm
    l2sum_row = tf.reduce_sum(values * values, axis=-1)
    pred = l2sum_row > 0
    l2sum_row_safe = tf.where(pred, l2sum_row, tf.ones_like(l2sum_row))
    l2norm_row = tf.sqrt(l2sum_row_safe)
    intermediate = values * tf.expand_dims(clip_t, -1)
    g_clip = intermediate / tf.expand_dims(tf.maximum(l2norm_row, clip_t), -1)

    if isinstance(g, tf.IndexedSlices):
        return tf.IndexedSlices(g_clip, g.indices, g.dense_shape)
    else:
        return g_clip
