# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # TF2


class GaussianConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), strides=(2, 2), padding="SAME", trainable=False, name=None, **kwargs):
        super(GaussianConv2DTranspose, self).__init__(name=name, trainable=trainable, **kwargs)
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = padding

    def build_filter1d(self, kernel_size):
        # Generate Pascal's triangle.
        # The final row of the triangle is an integer approximation of the normal distribution.
        # See: http://www.cse.psu.edu/~rtc12/CSE486/lecture10.pdf for more details.
        if kernel_size == 1:
            filter1d = [1]
        else:
            triangle = [[1, 1]]
            for i in range(1, kernel_size - 1):
                cur_row = [1]
                prev_row = triangle[i - 1]
                for j in range(len(prev_row) - 1):
                    cur_row.append(prev_row[j] + prev_row[j + 1])
                cur_row.append(1)
                triangle.append(cur_row)
            filter1d = triangle[-1]
        filter1d = tf.reshape(tf.convert_to_tensor(filter1d, tf.float32), kernel_size)
        return filter1d

    def build_weight(self, kernel_size, strides, num_in_channels):
        filter1d = self.build_filter1d(kernel_size=max(kernel_size[0], kernel_size[1]))
        if kernel_size[0] == kernel_size[1]:
            filter2d = tf.expand_dims(filter1d, 0) * tf.expand_dims(filter1d, -1)
            kernal = filter2d / tf.math.reduce_sum(filter2d)
            kernal = tf.expand_dims(tf.expand_dims(kernal, 0), 0)
            kernal = tf.tile(kernal, [num_in_channels, num_in_channels, 1, 1])
            kernal = tf.transpose(kernal, [2, 3, 0, 1])
            kernal = (kernal * tf.eye(num_in_channels))
            return kernal * float((strides[0] * strides[1]))
        if kernel_size[0] == 1:
            filter1d = tf.expand_dims(filter1d, 0)
            kernal = filter1d / tf.math.reduce_sum(filter1d)
            kernal = tf.expand_dims(tf.expand_dims(kernal, 0), 0)
            kernal = tf.tile(kernal, [num_in_channels, num_in_channels, 1, 1])
            kernal = tf.transpose(kernal, [2, 3, 0, 1])
            kernal = (kernal * tf.eye(num_in_channels))
            return kernal * float(max(strides[0], strides[1]))
        filter1d = tf.expand_dims(filter1d, -1)
        kernal = filter1d / tf.math.reduce_sum(filter1d)
        kernal = tf.expand_dims(tf.expand_dims(kernal, 0), 0)
        kernal = tf.tile(kernal, [num_in_channels, num_in_channels, 1, 1])
        kernal = tf.transpose(kernal, [2, 3, 0, 1])
        kernal = (kernal * tf.eye(num_in_channels))
        return kernal * float(max(strides[0], strides[1]))

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.channels = input_shape[3]
        else:
            self.channels = input_shape[3].value

        def kernel_initializer(_, dtype=tf.float32):
            return tf.cast(
                self.build_weight(kernel_size=self.kernel_size, strides=self.strides, num_in_channels=self.channels),
                dtype=dtype)

        self.conv = tf.keras.layers.Conv2DTranspose(self.channels, kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    use_bias=False,
                                                    trainable=False, kernel_initializer=kernel_initializer,
                                                    padding=self.padding)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.conv(inputs)

    def get_config(self):
        return {"strides": self.strides, "kernel_size": self.kernel_size, "padding": self.padding}


class GaussianUpsampling(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, smooth=True, trainable=False, name=None, **kwargs):
        super(GaussianUpsampling, self).__init__(name=name, trainable=trainable, **kwargs)
        self.strides = scale_factor
        self.kernel_size = scale_factor + scale_factor if smooth is True else scale_factor + scale_factor - 1
        self.conv1n = GaussianConv2DTranspose(kernel_size=(1, self.kernel_size), strides=(1, self.strides),
                                              padding="SAME")
        self.convn1 = GaussianConv2DTranspose(kernel_size=(self.kernel_size, 1), strides=(self.strides, 1),
                                              padding="SAME")

    def build(self, input_shape):
        if isinstance(input_shape[3], int):
            self.channels = input_shape[3]
            self.height = input_shape[1]
            self.width = input_shape[2]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        ones = tf.ones((1, self.height, self.width, self.channels), dtype=tf.float32)
        self.norm = tf.math.reduce_mean(self.conv1n(self.convn1(ones)), axis=-1, keepdims=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv1n(inputs)
        x = self.convn1(x)
        return x / self.norm

    def get_config(self):
        return {"scale_factor": self.scale_factor}


def BilinearSampling(tensor, scale_factor=2.0):
    if scale_factor == 1.0:
        return tensor
    height, width = tensor.get_shape().as_list()[1:3]
    method = 'bilinear'
    out = tf.image.resize(tensor, (int(height * scale_factor), int(width * scale_factor)), method=method)
    if out.dtype != tensor.dtype:
        out = tf.cast(out, tensor.dtype)
    return out


def CompareMethods(image, scale_factor):
    downsample = BilinearSampling(image, 1.0 / scale_factor)
    bilinear = BilinearSampling(downsample, scale_factor=scale_factor)
    gaussian = GaussianUpsampling(scale_factor=scale_factor, smooth=True)(downsample)
    bilinear_error = tf.math.reduce_sum(tf.math.abs(bilinear - image))
    gaussian_error = tf.math.reduce_sum(tf.math.abs(gaussian - image))
    return bilinear_error, gaussian_error


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.compat.v1.enable_eager_execution()
    image_filename = "image.jpg"  # download form https://thispersondoesnotexist.com/
    image = tf.cast(tf.expand_dims(tf.image.decode_image(tf.io.read_file(image_filename), channels=3), 0),
                    tf.float32) / 255.0
    scale_factor = 2
    bilinear_error, gaussian_error = CompareMethods(image, scale_factor)
    print("scale_factor:[", scale_factor, "]", "bilinear_error:", bilinear_error.numpy(), "gaussian_error:",
          gaussian_error.numpy())


if __name__ == "__main__":
    main()
