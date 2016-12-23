"""
A module that constructs anchor positions. Given k anchor boxes with different
widths and heights, we want to slide those anchors across every position of the
input feature map and output the coordinates of all these anchors.
Note that this module does not actually use the input (only its size) so its
backward pass always computes zero.
The constructor takes the following arguments:
- x0, y0: Numbers giving coordinates of receptive field centers for upper left
  corner of inputs.
- sx, sy: Numbers giving horizontal and vertical stride between receptive field
  centers.
- anchors: Tensor of shape (2, k) giving width and height for each of k anchor
  boxes.
Input:
N x C x H x W array of features
Output:
N x 4k x H x W array of anchor positions; if you reshape the output to
N x k x 4 x H x W then along the 3rd dim we have (xc, yc, w, h) giving the
coordinates of the anchor box at that location.
"""

import keras.backend as K
import numpy as np

# Hardcode field centers, just for tests
x0, y0, sx, sy = 1, 1, 10, 10

anchors = np.array([[45, 90], [90, 45], [64, 64],
                   [90, 180], [180, 90], [128, 128],
                   [181, 362], [362, 181], [256, 256],
                    [362, 724], [724, 362], [512, 512]]).T

num_anchors = anchors.shape[1]

MakeAnchorsParams = {
    'x0': x0,
    'y0': y0,
    'sx': sx,
    'sy': sy,
    'anchors': anchors,
    'N': 1,  # process one image in a time
    # 'W': conv_output.shape[0],
    # 'H': conv_output.shape[1],
    'k': num_anchors
}
setattr(K, 'params', MakeAnchorsParams)


def make_anchors(input_x):

    input_shape = input_x.get_shape().as_list()
    N, H, W = input_shape[0], input_shape[1], input_shape[2]
    x_centers = np.arange(0, W) * K.params['sx'] + K.params['x0']
    y_centers = np.arange(0, H) * K.params['sx'] + K.params['y0']

    output_np = np.zeros(shape=(N, 4 * K.params['k'], H, W))
    output_view = output_np.reshape(N, K.params['k'], 4, H, W)

    xc = output_view[:, :, 0, ...]
    yc = output_view[:, :, 1, ...]
    w = output_view[:, :, 2, ...]
    h = output_view[:, :, 3, ...]

    output_np[:, 0:12, ...] = np.tile(x_centers.reshape(1, 1, W), [K.params['k'], H, 1])
    output_np[:, 12:24, ...] = np.tile(y_centers.reshape(1, H, 1), [K.params['k'], 1, W])
    output_np[:, 24:36, ...] = np.tile(anchors[0].reshape(K.params['k'], 1, 1), [1, H, W])
    output_np[:, 36:48, ...] = np.tile(anchors[1].reshape(K.params['k'], 1, 1), [1, H, W])

    return K.variable(output_np)
