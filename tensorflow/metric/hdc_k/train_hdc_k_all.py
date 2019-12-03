# pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import math
import argparse
import time
import datetime
import numpy as np
import PIL.Image
import itertools
import signal
import traceback
import glob
import pprint
import tensorflow as tf
from tensorflow.python.client import device_lib

common_path = os.path.abspath( os.path.join(os.path.dirname(__file__), '../../') )
sys.path.insert(0, common_path ) if common_path not in sys.path else None

import metric_hdc_k_all as metric_learning
import common.dataset.sop as sop
from common.utils.miscs import ExponentialMovingAverage as EMA

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_optimizer(args, global_step):
    base_learning_rate = args.learning_rate
    lr_decay_rate = args.lr_decay_rate
    lr_decay_steps = args.lr_decay_steps
    lr_warmup_steps = args.lr_warmup_steps
    lr_warmup_ratio = args.lr_warmup_ratio
    lr_decay_staircase = args.lr_decay_staircase

    learning_rate = tf.constant(base_learning_rate)
    if lr_decay_rate < 1.0:
        learning_rate = tf.train.exponential_decay(
                base_learning_rate, global_step,
                lr_decay_steps, lr_decay_rate, staircase=lr_decay_staircase )
    if lr_warmup_steps > 0 :
        linear_warmup = lr_warmup_ratio + float(1.0- lr_warmup_ratio ) * tf.cast(global_step,tf.float32) / float(lr_warmup_steps)
        learning_rate = learning_rate * tf.clip_by_value( linear_warmup , lr_warmup_ratio , 1.0 )

    if args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate, args.beta1, args.beta2 )
    elif args.optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate, args.beta1, use_nesterov=True )
    elif args.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        assert None, "no optimizer for " + args.optimizer

    return optimizer


def create_multigpu_loss_from_networks(args, optimizer, endpoints_list, labels):
    nonzero_only = args.nonzero_only
    negative_margin = args.negative_margin
    positive_weight = args.positive_weight
    n_heads = args.n_heads
    sample_normalization = args.sample_normalization
    flat_hardratio = args.flat_hardratio

    #flat_hardratio = 0.6 works well
    hard_ratios = [flat_hardratio]*n_heads if flat_hardratio is not None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    if len(hard_ratios) < n_heads:
        hard_ratios += [ flat_hardratio ] * (n_heads - len(hard_ratios) )

    batch_labels = labels
    losses = []
    for j in range(n_heads):
        embedding = endpoints_list['embedding%d'%j]
        if sample_normalization :
            loss = metric_learning.hdc_one_loss_sn(embedding, batch_labels, hard_ratios[j], negative_margin, positive_weight )
        else:
            loss = metric_learning.hdc_one_loss(embedding, batch_labels, hard_ratios[j], negative_margin, positive_weight )
        losses.append( loss )
    loss_sum = tf.add_n( losses )
    if((tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))!=[]):
        loss_sum = loss_sum + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss_sum )
    # Calculate the gradients for the batch of data on this tower.
    grads = optimizer.compute_gradients(loss_sum)
    #######################################################################
    #######################################################################
    return grads

# def average_gradients(tower_grads):
#     """Calculate the average gradient for each shared variable across all towers.
#
#     Note that this function provides a synchronization point across all towers.
#
#     Args:
#     tower_grads: List of lists of (gradient, variable) tuples. The outer list
#       is over individual gradients. The inner list is over the gradient
#       calculation for each tower.
#     Returns:
#      List of pairs of (gradient, variable) where the gradient has been averaged
#      across all towers.
#     """
#     average_grads = []
#     for grad_and_vars in zip(*tower_grads):
#         # Note that each grad_and_vars looks like the following:
#         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#         grads = []
#         for g, _ in grad_and_vars:
#             # Add 0 dimension to the gradients to represent the tower.
#             expanded_g = tf.expand_dims(g, 0)
#
#             # Append on a 'tower' dimension which we will average over below.
#             grads.append(expanded_g)
#
#         # Average over the 'tower' dimension.
#         grad = tf.concat(axis=0, values=grads)
#         grad = tf.reduce_mean(grad, 0)
#
#         # Keep in mind that the Variables are redundant because they are shared
#         # across towers. So .. we will just return the first tower's pointer to
#         # the Variable.
#         v = grad_and_vars[0][1]
#         grad_and_var = (grad, v)
#         average_grads.append(grad_and_var)
#     return average_grads

def train(args):
    data_path = args.data_path
    data_path_base = args.data_path_base
    weight_path = args.weight_path
    save_path = args.save_path
    batch_size = args.batch_size
    embedding_dims = args.embedding_dims
    n_heads = args.n_heads
    positive_probability = args.positive_probability
    weighted_class_sampling = args.weighted_class_sampling
    base_network = args.base_network
    pooling = args.pooling
    weight_decay = args.weight_decay
    uniform_bias = args.uniform_bias
    augmentation = args.augmentation

    save_step = args.save_step
    learning_rate = args.learning_rate
    training_iters = args.training_iters
    training_iters = training_iters if training_iters >0 else sys.maxsize

    runtime_stats = args.runtime_stats
    write_summary = args.write_summary


    # Load the data before bulding the batch
    print("Loading data from", data_path, data_path_base)
    sampler = sop.SimpleClassSampler(data_path, positive_probability, weighted_class_sampling, data_path_base)
    print("Done.")

    DATA_IMAGE_SIZE = 256
    NET_INPUT_SIZE = 224
    if augmentation == 'scale':
        DATA_IMAGE_SIZE = 404

    ############################################################################
    with tf.device('/cpu:0'):
        phase_is_train = tf.placeholder_with_default(tf.constant(True), shape=[], name='phase_is_train')
        global_step = tf.Variable( 0, trainable=False, name='global_step',dtype=tf.int64 )
        gstep_inc = tf.assign(global_step, global_step+1)

        with tf.name_scope('opt') as scope:
            optimizer = get_optimizer(args, global_step)

        with tf.variable_scope(tf.get_variable_scope()):

            image_loader = sop.SquareScaleImageLoader(DATA_IMAGE_SIZE)
            batch_inputs  = sop.create_batch_by_sampling(
                    sampler, image_loader, DATA_IMAGE_SIZE,
                    batch_size=batch_size,
                    num_threads=8,
                    sampling_size=batch_size,
                    capacity= batch_size*2
                    )

            batch_imgs, batch_labels = batch_inputs
            with tf.device('/cpu:0'):
                endpoints, network = metric_learning.create_network(batch_imgs, base_network, pooling, augmentation, phase_is_train, DATA_IMAGE_SIZE, NET_INPUT_SIZE, embedding_dims, n_heads, uniform_bias, weight_decay)
                fc_embedding = endpoints['fc_embedding']
            grads = create_multigpu_loss_from_networks(args, optimizer, endpoints, batch_labels)

        # with tf.name_scope('grad_avg') as scope:
        #     # We must calculate the mean of each gradient. Note that this is the
        #     # synchronization point across all towers.
        #     if len(tower_grads) > 1:
        #         grads = average_gradients(tower_grads)
        #     else:
        #         grads = tower_grads[0]

        with tf.name_scope('grad_apply') as scope:
            # Apply the gradients to adjust the shared variables.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer_op  = optimizer.apply_gradients(grads)
            # should not pass the globa_step because it automatically increment it

        with tf.name_scope('loss_mean') as scope:
            loss_mean = tf.reduce_mean(tf.get_collection(tf.GraphKeys.LOSSES))
            tf.summary.scalar('loss_mean', loss_mean)

        with tf.name_scope('summary') as scope:
            summary_op = tf.summary.merge_all()

        print('Var list')
        var_list = list(filter(lambda x: (('InceptionV1' in x.name) or ('resnet' in x.name)) and ('Momentum' not in x.name), tf.global_variables()))
        for var in var_list:
            print(var.name)
        print('------------------------------------------------------')
        print('Global Variables')
        for var in tf.global_variables():
            print(var.name)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    ############################################################################

    tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False )

    ############################################################################
    # Load the pretrained weights and save
    if args.initialize:
        init = tf.global_variables_initializer() # tf.initialize_all_variables() is deprecated
        with tf.Session(config = tf_config ) as sess:
            print("Initializing the varaibles.")
            sess.run(init)
            print("loading. '%s'..."%weight_path)
            if(base_network == 'inceptionv1'):
                network.load( weight_path, sess, ignore_missing=True )
            else:
                saver_load = tf.train.Saver(var_list, max_to_keep=0)
                saver_load.restore(sess, weight_path)
            print("Saving '%s'..."%save_path)
            saver.save(sess, save_path, global_step=global_step.eval(sess) )
            summary_writer = tf.summary.FileWriter(save_path, sess.graph)
            summary_writer.flush()
            print("done.")
        # Done!
        return
    ############################################################################
    with tf.Session(config = tf_config ) as sess:
        if write_summary or runtime_stats:
            summary_writer = tf.summary.FileWriter(save_path, sess.graph)
        last_save_path = tf.train.latest_checkpoint( os.path.dirname(save_path) )
        print("Restoring from '%s'..."%last_save_path)
        saver.restore(sess, last_save_path )
        global_step_val =global_step.eval(sess)
        print('Starting from the step %d'%global_step_val)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        elapsed_avg = EMA()

        try:
            while global_step_val < training_iters :
                if coord.should_stop():
                    break
                be = time.time()

                if runtime_stats and global_step_val % 1000 == 10 :
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    loss_value, _ = sess.run([loss_mean, optimizer_op],
                            options=run_options, run_metadata=run_metadata )
                    summary_writer.add_run_metadata(run_metadata, 'step%d' % global_step_val , global_step_val )
                    print("Writing the runtime statistics!")
                    summary_writer.flush()
                else:
                    loss_value, _ = sess.run([loss_mean, optimizer_op])

                global_step_val = sess.run(gstep_inc)

                elapsed = time.time() - be
                elapsed_avg += elapsed

                print("{} Iter {}: Training Loss = {:0.6f} elapsed: {:02.1f}, avg: {:02.1f} ".format(
                        datetime.datetime.now(), global_step_val, loss_value, elapsed, elapsed_avg.average() )
                    )

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if write_summary and global_step_val % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step_val )

                if global_step_val %save_step == 0:
                    print("Saving '%s', global_step %d..."%( save_path, global_step_val ) )
                    saver.save(sess, save_path, global_step=global_step_val  )
        except Exception as ex:
            # Report exceptions to the coordinator.
            coord.request_stop(ex)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # Terminate as usual. It is safe to call `coord.request_stop()` twice.
            coord.request_stop()
            coord.join(threads)

        print("Finish!")
    ############################################################################

def main():
    parser = argparse.ArgumentParser()
    # Path & Initialization Options
    parser.add_argument('-s','--save_path', help='checkpoint path', required=True )
    parser.add_argument('-i','--initialize', help='make the initail checkpoint', action='store_true')
    parser.add_argument('-d','--data_path', help='input_path', default="Stanford_Online_Products/Ebay_train.txt" )
    parser.add_argument('--data_path_base', help='input_path root folder for images', default=None )
    parser.add_argument('-w','--weight_path', help='weight path', default='../../common/models/inception_v1.ckpt' )
    parser.add_argument('-t','--save_step', help='save checkpoint per N global steps, default=1000', type=int, default=1000 )
    # Preprocessing
    parser.add_argument('--augmentation', help='augmentation(crop,rotate,scale), default=crop', default='crop' )
    parser.add_argument('--positive_probability', help='positive_probability, default=1.0', type=float, default=1.0 )
    # Optimization Options
    parser.add_argument('-b','--batch_size', help='default=128', type=int, default=128 )
    parser.add_argument('-e','--training_iters', help='# of iteration, 0 means infinity, default=0', type=int, default=0 )
    parser.add_argument('-l','--learning_rate', help='learning_rate, default=0.01', type=float, default=0.01 )
    parser.add_argument('--lr_decay_rate', help='learning_rate exponential decay rate, default=0.5', type=float, default=0.5 )
    parser.add_argument('--lr_decay_steps', help='learning_rate exponential decay steps, default=10000', type=int, default=10000 )
    parser.add_argument('--lr_decay_staircase', help='staircase of exponential decay steps', action="store_true" )
    parser.add_argument('--lr_warmup_ratio', help='learning_rate multiplied by the value linearly increased from this to 1.0, default=0.25', type=float, default=0.25 )
    parser.add_argument('--lr_warmup_steps', help='learning_rate linearly increased until this iteration, default=0', type=int, default=0 )
    parser.add_argument('--weight_decay', help='weight decay, default=0.', type=float, default=0. )
    parser.add_argument('-o','--optimizer', help='optimizer[sgd,adam,momentum], default=momentum', default='momentum')
    parser.add_argument('--beta1', help='beta1 for adam, momentum, default=0.9', type=float, default=0.9)
    parser.add_argument('--beta2', help='beta2 for adam, default=0.999', type=float, default=0.999)
    # Loss options
    parser.add_argument('-n','--embedding_dims', help='embedding dimension, default=1024', type=int, default=1024 )
    parser.add_argument('--n_heads', help='# of heads for HDC, default=5', type=int, default=5)
    parser.add_argument('--uniform_bias', help='use uniform bias', action='store_true' )
    parser.add_argument('-m','--negative_margin', help='margin of the negative pair, default=0.5', type=float, default=0.5)
    parser.add_argument('--positive_weight', help="weight of positive pairs' loss, default=0.5", type=float, default='0.5')
    parser.add_argument('-z','--nonzero_only', help='make the average loss of non-zero negative only', action='store_true')
    parser.add_argument('--sample_normalization', help='the sample normalizatino loss ', action='store_true')
    parser.add_argument('--weighted_class_sampling', help='samples the classes with weights as number of image in that class', action='store_true', default=False)
    parser.add_argument('-f','--flat_hardratio', help='when specified, use same hard ratio for all losses', type=float, default=None)
    # inceptionv1: Uses Kaffe, [weight_decay and batchnorm updated] not implemented
    # inceptionv1bn, resnetv1_50: Uses Slim, [uniform_bias] not implemented
    parser.add_argument('--base_network', help='resnetv1_50, inceptionv1, inceptionv1bn', default='inceptionv1bn')
    parser.add_argument('--pooling', help='avg, avgnmax', default='avg')
    # Misc Options
    parser.add_argument('--write_summary', help='write the summary statistics', action='store_true')
    parser.add_argument('--runtime_stats', help='write the runtime statistics', action='store_true')
    args = parser.parse_args()
    pprint.pprint(vars(args))

    train(args)

if __name__ == "__main__":
    main()
