"""
Apply adjustments to bounding boxes for bounding box regression, with
backpropagation both into box offsets and box positions.

We use the same parameterization for box regression as R-CNN:
Given a bounding box with center (xa, ya), width wa, and height ha,
and given offsets (tx, ty, tw, th), we compute the new bounding box
(x, y, w, h) as:
    x = tx * wa + xa
    y = ty * ha + ya
    w = wa * exp(tw)
    h = ha * exp(th)

This parameterization is nice because the identity transform is (0, 0, 0, 0).

Module input: A list of
- boxes: Tensor of shape (D1, D2, ..., 4) giving coordinates of boxes in
         (xc, yc, w, h) format.
- trans: Tensor of shape (D1, D2, ..., 4) giving box transformations in the form
         (tx, ty, tw, th)

Module output:
- Tensor of shape (D1, D2, ..., 4) giving coordinates of transformed boxes in
  (xc, yc, w, h) format. Output has same shape as input.

"""
import tensorflow as tf
from keras.engine.topology import Layer


class ApplyBoxTransform(Layer):
    def __init__(self, **kwargs):
        super(ApplyBoxTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ApplyBoxTransform, self).build(input_shape)

    def call(self, input_x, mask=None):
        boxes, trans = input_x[0], input_x[1]

        d1, d2 = boxes.get_shape().as_list()

        xa, ya, wa, ha = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        tx, ty, tw, th = trans[:, 0], trans[:, 1], trans[:, 2], trans[:, 3]

        x = tf.multiply(tx, wa) + xa
        y = tf.multiply(tx, ha) + ya
        w = tf.multiply(wa, tf.exp(wa))
        h = tf.multiply(ha, tf.exp(th))

        output = tf.reshape(tf.stack([x, y, w, h]), [d1, d2])

        return output

    def get_output_shape_for(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]

        return (rows, cols)