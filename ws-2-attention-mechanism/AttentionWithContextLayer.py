from keras.layers import Layer

# from keras.engine import InputSpec
# from tensorflow.keras.layers import InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints


def dot_product(x, kernel):
    if K.backend() == "tensorflow":
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    def __init__(
        self,
        W_regularizer=None,
        u_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        u_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs
    ):

        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            name="{}_W".format(self.name),
            shape=(
                input_shape[-1],
                input_shape[-1],
            ),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )

        if self.bias:
            self.b = self.add_weight(
                name="{}_b".format(self.name),
                shape=(input_shape[-1],),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )

        self.u = self.add_weight(
            name="{}_u".format(self.name),
            shape=(input_shape[-1],),
            initializer=self.init,
            regularizer=self.u_regularizer,
            constraint=self.u_constraint,
        )

        super(AttentionWithContext, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
                'W_regularizer': self.W_regularizer,
                'u_regularizer': self.u_regularizer,
                'b_regularizer': self.b_regularizer,
                'W_constraint': self.W_constraint,
                'u_constraint': self.u_constraint,
                'b_constraint': self.b_constraint,
                'bias': self.bias,
        })
        return config

    def call(self, x):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.softmax(ait)
        a = K.expand_dims(a)
        weighted_input = x * a
        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
