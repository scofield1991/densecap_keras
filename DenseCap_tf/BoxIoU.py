
import tensorflow as tf


def xcycwh_to_x1y1x2y2(boxes):
    xc, yc, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    x0 = tf.div(tf.add(w, -1), 2.0) * (-1) + xc
    x1 = tf.div(tf.add(w, -1), 2.0) + xc
    y0 = tf.div(tf.add(h, -1), 2.0) * (-1) + yc
    y1 = tf.div(tf.add(h, -1), 2.0) + yc

    return tf.stack([x0, y0, x1, y1], axis=2)


def IoU(input_x):
    box_1 = input_x[0]
    box_2 = input_x[1]

    box_1_shape = box_1.get_shape().as_list()
    box_2_shape = box_2.get_shape().as_list()
    N, B1, B2 = box_1_shape[0], box_1_shape[1], box_2_shape[1]

    area_1 = box_1[:, :, 2] * box_1[:, :, 3]
    area_2 = box_2[:, :, 2] * box_2[:, :, 3]
    area_1_reshaped = tf.reshape(area_1, [N, B1, 1])
    area_2_reshaped = tf.reshape(area_2, [N, 1, B2])
    area_1_expand = tf.tile(area_1_reshaped, [1, 1, B2])
    area_2_expand = tf.tile(area_2_reshaped, [1, B1, 1])

    box1_lohi = xcycwh_to_x1y1x2y2(box_1)
    box2_lohi = xcycwh_to_x1y1x2y2(box_2)
    box1_lohi_reshaped = tf.reshape(box1_lohi, [N, B1, 1, 4])
    box2_lohi_reshaped = tf.reshape(box2_lohi, [N, 1, B2, 4])
    box1_lohi_expand = tf.tile(box1_lohi_reshaped, [1, 1, B2, 1])
    box2_lohi_expand = tf.tile(box2_lohi_reshaped, [1, B1, 1, 1])

    x0 = tf.maximum(box1_lohi_expand[..., 0], box2_lohi_expand[..., 0])
    y0 = tf.maximum(box1_lohi_expand[..., 1], box2_lohi_expand[..., 1])
    x1 = tf.minimum(box1_lohi_expand[..., 2], box2_lohi_expand[..., 2])
    y1 = tf.minimum(box1_lohi_expand[..., 3], box2_lohi_expand[..., 3])

    w = tf.maximum(x1 - x0, 0)
    h = tf.maximum(y1 - y0, 0)



    intersection = w * h
    output = tf.add(area_1_expand, (intersection * (-1)))
    output = tf.pow(tf.add(output, area_2_expand), -1) * intersection

    return output
