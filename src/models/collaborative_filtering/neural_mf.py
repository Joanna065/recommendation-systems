import tensorflow as tf
from official.recommendation import neumf_model


def create_model(params):
    user_input = tf.keras.layers.Input(shape=(1,),
                                       batch_size=None,
                                       name='user_id',
                                       dtype=tf.int32)

    item_input = tf.keras.layers.Input(shape=(1,),
                                       batch_size=None,
                                       name='movie_id',
                                       dtype=tf.int32)

    base_model = neumf_model.construct_model(user_input, item_input, params)
    rating = base_model.output

    return tf.keras.Model(inputs=[user_input, item_input], outputs=rating)


class Precision(tf.keras.metrics.Metric):
    def __init__(self, threshold=3.5, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        self.checker = tf.constant(2.0, dtype=tf.float32)
        self.tp = self.add_weight(name='true_positive', initializer='zeros')
        self.tp_fp = self.add_weight(name='recommended', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        relevant = tf.cast(y_true >= self.threshold, dtype=tf.float32)
        recommended = tf.cast(y_pred >= self.threshold, dtype=tf.float32)

        relevant_recommend = tf.reduce_sum(tf.cast(tf.reshape(
            relevant + recommended, (-1, 1)) >= self.checker, dtype=tf.float32))

        self.tp.assign_add(relevant_recommend)
        self.tp_fp.assign_add(tf.reduce_sum(recommended))

    def result(self):
        return tf.divide(self.tp, self.tp_fp)

    def reset_states(self):
        self.tp.assign(0.)
        self.tp_fp.assign(0.)


class Recall(tf.keras.metrics.Metric):
    def __init__(self, threshold=3.5, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        self.checker = tf.constant(2.0, dtype=tf.float32)
        self.tp = self.add_weight(name='true_positive', initializer='zeros')
        self.tp_fn = self.add_weight(name='relevant', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        relevant = tf.cast(y_true >= self.threshold, dtype=tf.float32)
        recommended = tf.cast(y_pred >= self.threshold, dtype=tf.float32)

        relevant_recommend = tf.reduce_sum(tf.cast(tf.reshape(
            relevant + recommended, (-1, 1)) >= self.checker, dtype=tf.float32))

        self.tp.assign_add(relevant_recommend)
        self.tp_fn.assign_add(tf.reduce_sum(relevant))

    def result(self):
        return tf.divide(self.tp, self.tp_fn)

    def reset_states(self):
        self.tp.assign(0.)
        self.tp_fn.assign(0.)


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, threshold=3.5, name='f1score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(threshold=threshold)
        self.recall = Recall(threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        prec = self.precision.result()
        rec = self.recall.result()
        return tf.multiply(2.0, tf.multiply(prec, rec) / tf.add(prec, rec))

    def reset_states(self):
        self.recall.reset_states()
        self.precision.reset_states()
