#!/usr/bin/python
"""
This is the metric learning using tensorflow

Author: keunjoo.kwon@samsung.com
Copyright (c) 2017 Samsung Electronics Co., Ltd. All Rights Reserved
"""
# pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import sys, os, math
import argparse
import time
import datetime
import numpy as np
import pickle
import PIL.Image
import itertools
import signal
import traceback
import glob
import tensorflow as tf
import gc

#export PYTHONPATH=$(pwd)/../../:${PYTHONPATH}
sys.path.insert(0, os.path.abspath('../../') )

import metric_hdc_k_all as metric_learning
#import common.dataset.preprocess as preprocess
import common.dataset.sop as sop
import common.utils.indexlib as indexlib
# import common.utils.gnumpy as gnumpy


np.set_printoptions(linewidth=120,precision=4,edgeitems=4)

###############################################################################
def group_generator(labels):
    cur_label = labels[0]
    grp_idxes = []
    for idx, label in enumerate(labels):
        if label != cur_label:
            yield np.array(grp_idxes)
            grp_idxes = []
        grp_idxes.append(idx)
        cur_label = label
    if len(grp_idxes) > 0:
        yield np.array(grp_idxes)

def group_generator_unordered(labels):
    data = itertools.izip( labels, range(len(labels) ) )
    data = sorted( data, key=lambda x: x[0] )
    for k, g in itertools.groupby(data, key=lambda x: x[0] ):
        _, idxs = zip(*g)
        yield np.array( list(idxs) )

def evaluate_all_ranks(features, labels):
    index = indexlib.BruteForceBLAS()
    index.fit(features)
    all_ranks = []
    for gidx, group in enumerate(group_generator_unordered(labels)):
        for i, query_id in enumerate(group):
            same_ids = group[np.arange(len(group))!=i]
            rank = index.check_rank( query_id, same_ids )
            all_ranks.append(rank)

        if gidx%100==0:
            sys.stdout.write('*')
    sys.stdout.write('\n')
    del index
    gc.collect()
    return np.array(all_ranks)


def extract_features(sess, fc_embedding, batch_labels):
    all_features, all_labels = [], []

    sess.run( tf.local_variables_initializer() ) # reset num_epochs to be 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for c in itertools.count():
            if coord.should_stop():
                print("coordinator stopping")
                break
            result, label = sess.run( [fc_embedding, batch_labels] )
            all_features.append(result)
            all_labels.append(label)
            sys.stdout.write('=')
    except tf.errors.OutOfRangeError:
        print('Done -- epoch limit reached')
    finally:
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sys.stdout.write('\n')

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    return features, labels

###############################################################################
def evaluate(args):
    data_path = args.data_path
    data_path_base = args.data_path_base
    save_path = args.save_path
    embedding_dims = args.embedding_dims
    batch_size = args.batch_size
    output = args.output
    loop_interval = args.loop_interval
    n_heads = args.n_heads
    base_network = args.base_network
    pooling = args.pooling
    augmentation = args.augmentation
    DATA_IMAGE_SIZE = 256
    NET_INPUT_SIZE = 224

    ##########################################################
    print("Loading data from", data_path, data_path_base)
    dataset = sop.GenericDataset(data_path, data_path_base)
    all_data = dataset()

    phase_is_train = tf.placeholder_with_default(tf.constant(False), shape=[], name='phase_is_train')
    image_loader = sop.SquareScaleImageLoader(DATA_IMAGE_SIZE)

    batch_inputs = sop.create_batch_by_slicing(all_data, image_loader,
            DATA_IMAGE_SIZE, batch_size, num_threads=8, num_epochs=1)
    batch_imgs, batch_labels = batch_inputs
    endpoints, network = metric_learning.create_network(batch_imgs, base_network, pooling, augmentation, phase_is_train, DATA_IMAGE_SIZE, NET_INPUT_SIZE, embedding_dims, n_heads)
    fc_embedding = endpoints['fc_embedding']
    print "Done."

    ##########################################################
    tfconfig = tf.ConfigProto( allow_soft_placement=True, log_device_placement=False )
    tfconfig.gpu_options.allow_growth=True
    saver = tf.train.Saver()
    ##########################################################
    all_results = dict()
    if os.path.exists( output ):
        print "Reading from '%s'..."%output
        all_results = pickle.load( open(output,'rb') )
    ##########################################################
    for loop in itertools.count():
        total_begin_time = time.time()
        checkpoint_files = sorted([ fpath.split('.')[0] for fpath in glob.glob( save_path + "-*.meta" ) ])
        ##########################################################
        for fpath in checkpoint_files :
            gstep_from_name = int( fpath.replace(save_path+'-','') )
            if gstep_from_name in all_results:
                continue
            with tf.Session(config=tfconfig) as sess: #should start each session because of closed_queue
                begin_time= time.time()
                ###############################################
                print "Restoring from '%s'..."%fpath
                saver.restore(sess, fpath )
                ###############################################
                features, labels = extract_features(sess, fc_embedding, batch_labels)
                print "All features are built!", features.shape, labels.shape,
                print "Elapsed %.1f secs."%( time.time() - begin_time )
                ###############################################
                begin_time= time.time()
                all_results[ gstep_from_name ] = evaluate_all_ranks(features, labels)
                print "Finished the evaluation. Elapsed %.1f secs."%( time.time() - begin_time )
                ###############################################
                pickle.dump( all_results, open(output,'wb'), -1 )
                print "Saved '%s' ..."%(output)
                ###############################################

        ##########################################################
        if loop_interval > 0 :
            print "Sleeping for another loop at {} ".format( datetime.datetime.now() )
            time.sleep( loop_interval )
        else:
            print "Total elapsed %.1f secs."%( time.time() - total_begin_time )
            break
    ##########################################################

##########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--save_path', help='checkpoint path', required=True )
    parser.add_argument('-o','--output', help='output file', required=True )
    parser.add_argument('-d','--data_path', help='input_path', required=True )
    parser.add_argument('--data_path_base', help='input_path root folder for images', default=None )
    parser.add_argument('--augmentation', help='augmentation(crop,rotate,scale), default=crop', default='crop' )
    parser.add_argument('-n','--embedding_dims', help='embedding dimension, default=1024', type=int, default=1024 )
    parser.add_argument('-b','--batch_size', help='default=128', type=int, default=128 )
    parser.add_argument('-l','--loop_interval', help='loop interval default=0', type=int, default=0 )
    parser.add_argument('-g','--gpu_id', help='gpu_id, default=0', type=int, default=0 )
    parser.add_argument('--n_heads', help='# of heads for HDC, default=5', type=int, default=5)
    parser.add_argument('--base_network', help='resnetv1_50, inceptionv1, inceptionv1bn', default='inceptionv1bn')
    parser.add_argument('--pooling', help='avg, avgnmax', default='avg')
    args = parser.parse_args()
    # gnumpy.board_id_to_use = args.gpu_id

    evaluate(args)

if __name__ == "__main__":
    main()
