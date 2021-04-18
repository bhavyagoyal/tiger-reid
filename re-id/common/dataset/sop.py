from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import PIL.Image
import os
import sys
import time
import itertools
import traceback
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# pylint: disable=line-too-long

##########################################################################
def create_metric_batch_by_slicing(dataset, image_loader, image_size,
                            batch_size=32, num_threads=1, shuffle=False, num_epochs=None, capacity=None):
    capacity = batch_size*num_threads if capacity is None else capacity

    # Must run tf.local_variables_initializer() to set num_epochs in slice_input_producer
    # otherewise, it will cause the "attempting to use uninitialized value" error
    data_queue = tf.train.slice_input_producer( dataset,
            num_epochs=num_epochs, shuffle=shuffle, capacity=capacity )

    data_records = tf.py_func(image_loader, data_queue, [tf.float32, tf.int32, tf.int32, tf.string] )

    images, labels, categories, image_paths = data_records
    images.set_shape( [None, image_size, image_size, 3] )
    labels.set_shape( [None] )
    categories.set_shape( [None] )
    image_paths.set_shape( [None] )

    batch_input = tf.train.batch(
            [images, labels, categories, image_paths] , batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True)

    return batch_input

def create_batch_by_slicing(*args, **kwds):
    batch_imgs, batch_labels, batch_categories, batch_image_paths = create_metric_batch_by_slicing(*args, **kwds)
    return batch_imgs, batch_labels

##########################################################################
# sampler -> image_file_queue -> load_image_func  -> batch_queue -> batch_input
# 1. Sample the image files and class ids of sample_size, without replication
#    Contains positive pairs with some probabilities
# 2. Dequeue one and load the image
def create_metric_batch_by_sampling(sampler, image_loader, image_size,
                            batch_size=32, num_threads=1, sampling_size=None, capacity=None):
    sampling_size = batch_size if sampling_size is None else sampling_size
    capacity = batch_size*num_threads if capacity is None else capacity

    image_file_queue = tf.FIFOQueue( capacity=sampling_size*2, dtypes=[tf.string, tf.int32, tf.int32] )
    image_file_sampler = tf.py_func( sampler, [sampling_size], [tf.string, tf.int32, tf.int32] )
    image_file_enqueue_op = image_file_queue.enqueue_many(image_file_sampler)

    tf.train.add_queue_runner( tf.train.QueueRunner(image_file_queue, [image_file_enqueue_op]) )

    data_records = tf.py_func(image_loader, image_file_queue.dequeue(), [tf.float32, tf.int32, tf.int32, tf.string] )
    images, labels, categories, image_paths = data_records
    images.set_shape( [None, image_size, image_size, 3] )
    labels.set_shape( [None] )
    categories.set_shape( [None] )
    image_paths.set_shape( [None] )

    batch_input = tf.train.batch(
            [images, labels, categories, image_paths] , batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True)
    return batch_input

def create_batch_by_sampling(*args, **kwds):
    batch_imgs, batch_labels, batch_categories, batch_image_paths = create_metric_batch_by_sampling(*args, **kwds)
    return batch_imgs, batch_labels

##########################################################################
# Preload the data from the image-list file
"""
    < Sample image-list file's format >
        image_id class_id super_class_id path
        1 1 1 bicycle_final/111085122871_0.JPG
        2 1 1 bicycle_final/111085122871_1.JPG
        3 1 1 bicycle_final/111085122871_2.JPG
        4 1 1 bicycle_final/111085122871_3.JPG
        5 2 1 bicycle_final/111265328556_0.JPG
        6 2 1 bicycle_final/111265328556_1.JPG
        7 2 1 bicycle_final/111265328556_2.JPG
        8 2 1 bicycle_final/111265328556_3.JPG
        9 2 1 bicycle_final/111265328556_4.JPG
"""
class GenericDataset(object):
    def __init__(self, data_path, data_path_base = None):
        self.load_input_file(data_path, data_path_base)

    def load_input_file(self, data_path, data_path_base):
        input_file = open(data_path, 'r').readlines()
        input_file = input_file[1:]
        # skip the first line
        self.class_N_filename_list = list()
        for line in input_file:
            image_id_str, class_id_str, super_class_id_str, filename = line.split(' ')[:4]
            self.class_N_filename_list.append( ( int(class_id_str), int(super_class_id_str), filename.strip()) )
        if(data_path_base is None):
            self.data_path_base = os.path.dirname( data_path )
        else:
            self.data_path_base = data_path_base

    def __call__(self):
        class_N_filename_list  = self.class_N_filename_list
        classes, super_classes, filenames = itertools.izip(*class_N_filename_list)
        filenames  = [ os.path.join( self.data_path_base, file_name ) for file_name in filenames  ]
        return [filenames, classes, super_classes ]

class MetricDataset(object):
    def __init__(self, data_path, data_path_base = None):
        self.load_input_file(data_path, data_path_base)

    def load_input_file(self, data_path, data_path_base):
        input_file = open(data_path, 'r').readlines()[1:]
        # skip the first line
        class_2_image_map = dict()
        class_2_category_map = dict()

        for no, line in enumerate(input_file):
            image_id_str, class_id_str, super_class_id_str, filename = line.split(' ')[:4]
            class_id = int(class_id_str)
            if class_id not in class_2_image_map:
                class_2_image_map[ class_id ] = list()
            class_2_image_map[ class_id ].append( filename.strip() )
            class_2_category_map[class_id] = int(super_class_id_str)

        self.class_N_images_list = list(class_2_image_map.items())  # the list of the lists
        self.class_2_category_map = class_2_category_map
        self.classidx_pool  = range( len( self.class_N_images_list ) )
        if(data_path_base is None):
            self.data_path_base = os.path.dirname( data_path )
        else:
            self.data_path_base = data_path_base

class SimpleClassSampler(MetricDataset):
    def __init__(self, data_path, positive_probability = 1.0, weighted_class_sampling = False, data_path_base = None):
        super(SimpleClassSampler,self).__init__(data_path, data_path_base)
        self.positive_probability = positive_probability
        self.weighted_class_sampling = weighted_class_sampling

    def __call__(self, sampling_size):
        """
        Sample the class ids of the sampling_size, without replication
        if sampling_size>#classes then we sample all classes
        """
        classidx_pool = self.classidx_pool
        data_path_base = self.data_path_base
        positive_probability = self.positive_probability
        weighted_class_sampling = self.weighted_class_sampling
        class_sampling_size = min(sampling_size, len(classidx_pool))
        if(weighted_class_sampling):
            weights = np.array([len(v) for k,v in self.class_N_images_list])
            weights = weights/sum(weights)
            sample_idxs = np.random.choice(classidx_pool, class_sampling_size, replace=False, p=weights) # sampling without replacement.
        else:
            sample_idxs = random.sample(classidx_pool, class_sampling_size) # sampling without replacement.
        class_N_images_list = self.class_N_images_list
        class_2_category_map = self.class_2_category_map

        image_file_list = []
        class_id_list = []
        category_id_list = []

        # Do sampling
        for idx in sample_idxs:
            N = 1
            if positive_probability <= 1.0:
                is_positive = random.random() < positive_probability
                N = 2 if is_positive else 1
            else:
                N = int(positive_probability)

            label, class_images = class_N_images_list[idx]
            N = min( N, len(class_images) )
            image_file_list += random.sample( class_images, N )
            class_id_list += [label]*N
            category_id_list += [ class_2_category_map[label] ]*N

        # truncate the size to be sampling_size
        image_file_list = image_file_list[:sampling_size]
        class_id_list = class_id_list [:sampling_size]
        category_id_list = category_id_list[:sampling_size]

        # Make arrays
        image_path_list = [ os.path.join( data_path_base, file_name ) for file_name in image_file_list ]
        image_path_array = np.char.asarray(image_path_list)

        class_id_array = np.array(class_id_list, dtype=np.int32)
        category_id_array = np.array(category_id_list, dtype=np.int32)

        return image_path_array, class_id_array, category_id_array

##########################################################################
def pil_square_scale(im, img_size, bgColor = '#FFFFFF' ):
    if im.size[0] == im.size[1]:
        nim = im
    else:
        nim = PIL.Image.new( im.mode, ( max(im.size),max(im.size) ), bgColor )
        x_offset = (max(im.size) - im.size[0]) //2
        y_offset = (max(im.size) - im.size[1]) //2
        nim.paste( im, (x_offset, y_offset) )

    nim2 = nim.resize( (img_size, img_size) , PIL.Image.BICUBIC)
    return nim2

class SquareScaleImageLoader(object):
    def __init__(self, target_image_size):
        PIL.Image.LANCZOS  # To check if the PILLOW version is higher than 2.7.0
        self.target_image_size = target_image_size

    def __call__(self, image_path, label, category):
        target_image_size = self.target_image_size
        try:
            im = PIL.Image.open( image_path )
            im = im.convert("RGB") if im.mode != "RGB" else im
            im = pil_square_scale(im, target_image_size )
            im_array = np.array( im ).astype( np.float32, copy=False )
            im_array = im_array.reshape( (1,) + im_array.shape )
        except Exception as ex:
            print (image_path, ':', ex)
            traceback.print_exc()
            im_array = np.zeros( (0, target_image_size, target_image_size, 3), dtype=np.float32 )
            label = np.zeros( (0,), dtype=np.int32 )
            category = np.zeros( (0,), dtype=np.int32 )
            image_path = np.char.asarray([])
        return im_array, np.array( [label], dtype=np.int32) if type(label) is not np.ndarray else label, \
               np.array( [category], dtype=np.int32) if type(category) is not np.ndarray else category, \
               np.char.asarray([image_path]) if type(image_path) is not np.core.defchararray.chararray else image_path
#np.array([image_path], dype=object)

class SquareStrechImageLoader(object):
    def __init__(self, target_image_size):
        PIL.Image.LANCZOS  # To check if the PILLOW version is higher than 2.7.0
        self.target_image_size = target_image_size

    def __call__(self, image_path, label, category):
        target_image_size = self.target_image_size
        try:
            im = PIL.Image.open( image_path )
            im = im.convert("RGB") if im.mode != "RGB" else im
            im = im.resize( (target_image_size , target_image_size ) , PIL.Image.BICUBIC)
            im_array = np.array( im ).astype( np.float32, copy=False )
            im_array = im_array.reshape( (1,) + im_array.shape )
        except Exception as ex:
            print (image_path, ':', ex)
            traceback.print_exc()
            im_array = np.zeros( (0, target_image_size, target_image_size, 3), dtype=np.float32 )
            label = np.zeros( (0,),dtype=np.int32 )
        return im_array, np.array( [label], dtype=np.int32), np.array( [category], dtype=np.int32), np.char.asarray([image_path])

##########################################################################
# Test Functions
def test_batch(is_train, data_path):
    s = tf.InteractiveSession()
    image_loader = SquareScaleImageLoader(256)

    if is_train:
        sampler = SimpleClassSampler(data_path, data_path_base='/home/bhavya/datasets/CUB/CUB_200_2011/images')
        batch = create_batch_by_sampling(sampler, image_loader, 256, 10, 8)
    else:
        dataset = GenericDataset(data_path)
        all_data = dataset()
        batch = create_batch_by_slicing(all_data, image_loader, 256, 10, 8, num_epochs=1)
        s.run( tf.local_variables_initializer() )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        for c in itertools.count():
            if coord.should_stop():
                print("stopping")
                break
            print('-------------')
            imgs, labels = s.run(batch)
            print (c, labels)
            #for i in range(len(imgs)):
                #print type(imgs[i] ), labels[i]
                #imgs_int = imgs[i].astype( np.uint8)
                #im = PIL.Image.fromarray(imgs_int )
                #im.show()

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
