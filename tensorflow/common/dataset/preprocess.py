"""
This is the preprocessing using tensorflow

Author: keunjoo.kwon@samsung.com
Copyright (c) 2017 Samsung Electronics Co., Ltd. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# pylint: disable=line-too-long

# use switchChannel=False, scalePixels = True, mean=128. for slim models
def shared_processing(input_images, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = tf.cast(input_images, tf.float32)
    imgs = (imgs-mean) if mean is not None else imgs
    imgs = imgs*2./255. if scalePixels else imgs
    imgs = tf.reverse(imgs, axis=[3]) if switchChannel else imgs
    return imgs


def random_crop(input_images, phase_is_train, image_size, input_size, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = shared_processing(input_images, switchChannel, scalePixels, mean)

    random_cropper = lambda i: tf.random_crop(i, [input_size, input_size, 3])
    offset = ( image_size  - input_size ) //2

    final_imgs  = tf.cond(phase_is_train,
            lambda: tf.map_fn(random_cropper, imgs),
            lambda: tf.slice( imgs, [0, offset, offset, 0], [-1, input_size, input_size, -1 ] )
        )
    return final_imgs

def random_crop_flip(input_images, phase_is_train, image_size, input_size, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = shared_processing(input_images, switchChannel, scalePixels, mean)

    random_cropper = lambda i: tf.random_crop(i, [input_size, input_size, 3])
    random_flipper = lambda i: tf.image.random_flip_left_right(i)
    offset = ( image_size  - input_size ) //2

    final_imgs  = tf.cond(phase_is_train,
            lambda: tf.map_fn( random_flipper, tf.map_fn(random_cropper, imgs) ),
            lambda: tf.slice( imgs, [0, offset, offset, 0], [-1, input_size, input_size, -1 ] )
        )
    return final_imgs


# Final out of image is not of input_size but of input_size*scale
# Used for training DELF with varying input size
def random_crop_flip_scale(input_images, phase_is_train, image_size, input_size, scales, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = shared_processing(input_images, switchChannel, scalePixels, mean)

    center_offset = ( image_size  - input_size ) //2
    test_transform = lambda: tf.slice( imgs, [0, center_offset, center_offset, 0], [-1, input_size, input_size, -1 ] )

    cropped_imgs = tf.map_fn(lambda i: tf.random_crop(i, [input_size, input_size, 3]), imgs )
    flipped_imgs = tf.map_fn(lambda i: tf.image.random_flip_left_right(i), cropped_imgs )

    scales_tensor = tf.convert_to_tensor(scales)
    sample_scale = tf.reshape( tf.multinomial( tf.expand_dims(tf.ones_like(scales_tensor), axis=0) , num_samples=1), shape=[])
    out_size = tf.gather(scales_tensor, sample_scale)*input_size
    output_size = tf.cast(tf.convert_to_tensor([out_size, out_size]), dtype=tf.int32)
    #output_size = tf.Print(output_size, [output_size], "output size; ")
    resize = tf.map_fn(lambda i: tf.image.resize_images(i, output_size, method=tf.image.ResizeMethod.AREA), flipped_imgs)
    training_transform = lambda: resize

    final_imgs  = tf.cond(phase_is_train, training_transform, test_transform)
    return final_imgs


def random_rotate_crop_flip(input_images, phase_is_train, image_size, input_size, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = shared_processing(input_images, switchChannel, scalePixels, mean)

    # Test preprocessing
    center_offset = ( image_size  - input_size ) //2
    test_transform = lambda: tf.slice( imgs, [0, center_offset, center_offset, 0], [-1, input_size, input_size, -1 ] )

    # Augmentation for training
    angles = tf.random_uniform( tf.shape(imgs) [:1] , -np.pi/6. , np.pi/6. ) # -30 ~ +30 degree
    rotated_imgs = tf.contrib.image.rotate( imgs, angles )
    cropped_imgs = tf.map_fn(lambda i: tf.random_crop(i, [input_size, input_size, 3]), rotated_imgs )
    flipped_imgs = tf.map_fn( lambda i: tf.image.random_flip_left_right(i), cropped_imgs )
    training_transform = lambda: flipped_imgs

    final_imgs  = tf.cond(phase_is_train, training_transform, test_transform )

    return final_imgs


def random_rotate_scale_crop_flip(input_images, phase_is_train, image_size, input_size, switchChannel = True, scalePixels = False, mean = tf.constant([123., 117., 104.])):
    imgs = shared_processing(input_images, switchChannel, scalePixels, mean)

    mean_size = ( input_size + image_size )//2
    center_offset = ( mean_size  - input_size ) //2
    # Test preprocessing
    test_scaling = tf.image.resize_images( imgs, [mean_size, mean_size] , method=tf.image.ResizeMethod.AREA )
    test_transform = lambda: tf.slice( test_scaling, [0, center_offset, center_offset, 0], [-1, input_size, input_size, -1 ] )

    # Augmentation for training
    angles = tf.random_uniform( tf.shape(imgs) [:1] , -np.pi/6. , np.pi/6. ) # -30 ~ +30 degree
    rotated_imgs = tf.contrib.image.rotate( imgs, angles )

    upper_left_max = ( image_size  - input_size ) //2
    lower_right_min= ( image_size  + input_size ) //2
    x_offsets = tf.random_uniform( tf.shape(imgs) [:1] , 0, upper_left_max+1, dtype=tf.int32 )
    y_offsets = tf.random_uniform( tf.shape(imgs) [:1] , 0, upper_left_max+1, dtype=tf.int32 )
    min_sizes = tf.maximum( lower_right_min - x_offsets , lower_right_min - y_offsets )
    max_sizes = tf.minimum( image_size - x_offsets , image_size - y_offsets ) + 1

    # val = (min_val, max_val)
    box_sizes = tf.map_fn(
        lambda val : tf.random_uniform([1], val[0], val[1], dtype=tf.int32 ) ,
            (min_sizes, max_sizes), dtype = tf.int32 )
    box_sizes= tf.reshape( box_sizes, [-1] )

    def crop_and_resize(params):
        img, y_offset, x_offset, target_size = params
        cropped = tf.image.crop_to_bounding_box( img, y_offset, x_offset, target_size, target_size )
        resized = tf.image.resize_images( cropped, [input_size, input_size], method=tf.image.ResizeMethod.AREA )
        return resized

    cropped_imgs = tf.map_fn( crop_and_resize, ( rotated_imgs, y_offsets, x_offsets, box_sizes ), dtype=tf.float32 )
    flipped_imgs = tf.map_fn( lambda i: tf.image.random_flip_left_right(i), cropped_imgs )
    training_transform = lambda: flipped_imgs

    final_imgs  = tf.cond(phase_is_train, training_transform, test_transform )

    return final_imgs
