"""
This is DataSet Loader for classification

"""
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
from sets import Set
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# pylint: disable=line-too-long

def create_batch_by_slicing(dataset, image_loader, image_size,
                            batch_size=32, num_threads=1, shuffle=False, num_epochs=None, capacity=None):
    capacity = batch_size*num_threads if capacity is None else capacity

    # Must run tf.local_variables_initializer() to set num_epochs in slice_input_producer
    # otherewise, it will cause the "attempting to use uninitialized value" error
    data_queue = tf.train.slice_input_producer( dataset,
            num_epochs=num_epochs, shuffle=shuffle, capacity=capacity )

    data_records = tf.py_func(image_loader, data_queue, [tf.float32, tf.int32, tf.string] )

    images, labels, image_paths = data_records 
    images.set_shape( [None, image_size, image_size, 3] )
    labels.set_shape( [None] )
    image_paths.set_shape( [None] )
    batch_imgs, batch_labels, _ = tf.train.batch(
            [images, labels, image_paths] , batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True)

    return batch_imgs, batch_labels 



def create_batch_by_sampling(sampler, image_loader, image_size,
                            batch_size=32, num_threads=1, capacity=None):
    print ('start create_batch_by_sampling')
    capacity = batch_size*num_threads if capacity is None else capacity

    image_file_queue = tf.FIFOQueue( capacity=batch_size*2, dtypes=[tf.string, tf.int32] )
    image_file_sampler = tf.py_func( sampler, [batch_size], [tf.string, tf.int32] )
    image_file_enqueue_op = image_file_queue.enqueue_many(image_file_sampler)

    tf.train.add_queue_runner( tf.train.QueueRunner(image_file_queue, [image_file_enqueue_op]) ) 

    data_records = tf.py_func(image_loader, image_file_queue.dequeue(), [tf.float32, tf.int32, tf.string] )
    images, labels, image_paths = data_records 
    images.set_shape( [None, image_size, image_size, 3] )
    labels.set_shape( [None] )
    image_paths.set_shape( [None] )

    batch_imgs, batch_labels, _ = tf.train.batch(
            [images, labels, image_paths] , batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True)

    return batch_imgs, batch_labels

    
##########################################################################
# Preload the data from the image-list file
"""
    < Sample image-list file's format >
        1 bicycle_final/111085122871_0.JPG
        1 bicycle_final/111085122871_1.JPG
        1 bicycle_final/111085122871_2.JPG
        1 bicycle_final/111085122871_3.JPG
        2 bicycle_final/111265328556_0.JPG
        2 bicycle_final/111265328556_1.JPG
        2 bicycle_final/111265328556_2.JPG
        2 bicycle_final/111265328556_3.JPG
        2 bicycle_final/111265328556_4.JPG
"""
class GenericDataset(object):
    def __init__(self, data_path, data_path_base = None, is_id_first = True):
        self.load_input_file(data_path, data_path_base, is_id_first)

    def load_input_file(self, data_path, data_path_base, is_id_first):
        input_file = open(data_path)
        #input_file.next() # skip the first line
        self.class_N_filename_list = list()
        for line in input_file:
            l = line.split(' ')
            if is_id_first:
                class_id_str, filename = l[0], ' '.join(l[1:])
            else:
                class_id_str, filename = l[-1].strip(), ' '.join(l[:-1])    
            self.class_N_filename_list.append( ( int(class_id_str), filename.strip()) )
        if(data_path_base is None):
            self.data_path_base = os.path.dirname( data_path )
        else:
            self.data_path_base = data_path_base

    def __call__(self):
        class_N_filename_list  = self.class_N_filename_list 
        classes, filenames = itertools.izip(*class_N_filename_list)
        filenames  = [ os.path.join( self.data_path_base, file_name ) for file_name in filenames  ]
        return [filenames, classes ]


class ClsDataset(object):
    def __init__(self, data_path, data_path_base = None, is_id_first = True):
        self.load_input_file(data_path, data_path_base, is_id_first)

    def load_input_file(self, data_path, data_path_base, is_id_first):
        input_file = open(data_path)
        #input_file.next() # skip the first line
        class_2_image_count = dict()
        self.class_N_filename_list = list()

        for no, line in enumerate(input_file):
            l = line.split(' ')
            if is_id_first:
                class_id_str, filename = l[0], ' '.join(l[1:])
            else:
                class_id_str, filename = l[-1].strip(), ' '.join(l[:-1])
            if no < 5:
                print ( class_id_str, filename )
            class_id = int(class_id_str)
            self.class_N_filename_list.append( ( class_id, filename.strip()) )
            if class_id not in class_2_image_count:
                class_2_image_count[ class_id ] = 0
            class_2_image_count[ class_id ]+=1
        num_classes = len(class_2_image_count.keys())
        self.weights = [1.0/(num_classes*class_2_image_count[x[0]]) for x in self.class_N_filename_list]

        if(data_path_base is None):
            self.data_path_base = os.path.dirname( data_path )
        else:
            self.data_path_base = data_path_base


class SimpleSampler(ClsDataset):
    def __init__(self, data_path, data_path_base = None, balanced_sampling = False, is_id_first = True):
        super(SimpleSampler,self).__init__(data_path, data_path_base, is_id_first)
        self.balanced_sampling = balanced_sampling

    def __call__(self, sampling_size):
        data_path_base = self.data_path_base
        balanced_sampling = self.balanced_sampling
        if(balanced_sampling):
            sample_idxs = np.random.choice(len(self.class_N_filename_list), sampling_size, replace=False, p=self.weights)
        else:
            sample_idxs = np.random.choice(len(self.class_N_filename_list), sampling_size, replace=False)
        samples = [self.class_N_filename_list[x] for x in sample_idxs]
        classes, filenames = itertools.izip(*samples)
        image_path_list = [ os.path.join( data_path_base, file_name ) for file_name in filenames ]

        image_path_array = np.char.asarray(image_path_list)
        class_id_array = np.array(classes, dtype=np.int32)
        return image_path_array, class_id_array 


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

    def __call__(self, image_path, label):
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
            image_path = np.char.asarray([])
        return im_array, np.array( [label], dtype=np.int32) if type(label) is not np.ndarray else label, \
               np.char.asarray([image_path]) if type(image_path) is not np.core.defchararray.chararray else image_path
#np.array([image_path], dype=object)

class SquareStrechImageLoader(object):
    def __init__(self, target_image_size):
        PIL.Image.LANCZOS  # To check if the PILLOW version is higher than 2.7.0  
        self.target_image_size = target_image_size

    def __call__(self, image_path, label):
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
        return im_array, np.array( [label], dtype=np.int32), np.char.asarray([image_path])

##########################################################################
# Test Functions
def test_batch(is_train = False, data_path='datatxt/shoes_v2/train_shoes.txt'):
    s = tf.InteractiveSession()
    image_loader = SquareScaleImageLoader(256)

    if is_train:
        sampler = SimpleSampler(data_path)
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

