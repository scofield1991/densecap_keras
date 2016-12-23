"""
Input a tensor of shape N x (D * k) x H x W
Reshape and permute to output a tensor of shape N x (k * H * W) x D
"""
import keras.backend as K
from keras.engine.topology import Layer


class ReshapeBoxFeatures(Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(ReshapeBoxFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeBoxFeatures, self).build(input_shape)

    def call(self, input_x, mask=None):
        H, W = input_x.get_shape().as_list()[2], input_x.get_shape().as_list()[3]
        D = input_x.get_shape().as_list()[1] / self.k
        k = self.k
        input_x = K.reshape(input_x, [k * H * W, D])
        # print (k, H, W, D)
        return input_x

    def get_output_shape_for(self, input_shape):
        D = input_shape[1] / self.k
        return (input_shape[1], self.k * input_shape[2] * input_shape[3], D)
