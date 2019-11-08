"""
This is the metric learning using tensorflow

Author: keunjoo.kwon@samsung.com
Copyright (c) 2017 Samsung Electronics Co., Ltd. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
import common.dataset.preprocess as preprocess
slim = tf.contrib.slim
from nets import inception_v1
from nets import resnet_v1


def create_network(batch_imgs, base_network, pooling, augmentation, phase_is_train, data_image_size, net_input_size, embedding_dims, n_heads, uniform_bias=False, weight_decay=0.):
    if(base_network=='inceptionv1bn'):
        switchChannel = False
        scalePixels = True
        mean = 128.
    elif(base_network=='resnetv1_50'):
        switchChannel = False
        scalePixels = False
        mean = tf.constant([123., 117., 104.])
    else:
        assert 0, 'Choose supported base architecture'

    if augmentation == 'crop':
        augmentation_fn = preprocess.random_crop_flip
    elif augmentation == 'rotate':
        augmentation_fn = preprocess.random_rotate_crop_flip
    elif augmentation == 'scale':
        augmentation_fn = preprocess.random_rotate_scale_crop_flip
    else:
        assert 0, 'Unknown augmentation %s'%augmentation

    processed_imgs = augmentation_fn(
                    batch_imgs, phase_is_train, data_image_size, net_input_size, switchChannel=switchChannel, scalePixels=scalePixels, mean=mean)
    if(base_network=='inceptionv1bn'):
        network = make_inceptionv1bn_multi_embeddings(processed_imgs, embedding_dims, n_heads, phase_is_train, uniform_bias, weight_decay, pooling)
    elif(base_network=='resnetv1_50'):
        network = make_resnetv1_50_multi_embeddings(processed_imgs, embedding_dims, n_heads, phase_is_train, uniform_bias, weight_decay)
    return network


# pylint: disable=line-too-long
def uniform_fc( network, prev_name, name, num_out, relu):
    with tf.variable_scope(name) as scope:
        prev = network.layers[prev_name]
        input_shape = prev.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(prev, [-1, dim])
        else:
            feed_in, dim = (prev, input_shape[-1].value)

        sc_initializer  = tf.uniform_unit_scaling_initializer(1.0)
        weights = tf.get_variable('weights', shape=[dim, num_out])
        biases = tf.get_variable('biases', [num_out], initializer=sc_initializer )
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        network.layers[name] = op(feed_in, weights, biases, name=scope.name)


# uniform bias not used
def make_resnetv1_50_multi_embeddings(batch_imgs, embedding_dims, n_heads, phase_is_train, uniform_bias=False, weight_decay=0.00004):
    blocks = 4
    units = [3,4,6,3]
    emb_info = ['resnet_v1_50/block1', 'resnet_v1_50/block2', 'resnet_v1_50/block3', 'resnet_v1_50/block4']
    if(n_heads==16):
       emb_info = []
       for i in xrange(blocks):
            for j in xrange(units[i]):
                emb_info.append('resnet_v1_50/block'+str(i+1)+'/unit_'+str(j+1)+'/bottleneck_v1')

    left_embedding = embedding_dims
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        net, endpoints = resnet_v1.resnet_v1_50(batch_imgs,
            num_classes=0,
            global_pool=False,
            is_training = phase_is_train
            )
        for i in range(n_heads):
            emb_dim = int( math.ceil(left_embedding / float(n_heads-i) ) )
            left_embedding -= emb_dim
            with tf.variable_scope('loss%d'%i) as scope:
		# change fully connected to conv2d for using Regularization losses of slim in resent args scope
                endpoints['emb_%d'%i] = slim.fully_connected( tf.reduce_mean(endpoints[emb_info[i]], [1,2]),
                     emb_dim,
                     activation_fn=None)
                endpoints['embedding%d'%i] = tf.nn.l2_normalize( endpoints['emb_%d'%i], dim=1)

        with tf.variable_scope('fc_embedding') as scope:
            embs = [ endpoints['embedding%d'%i] for i in range(n_heads) ]
            endpoints['fc_embedding'] = tf.concat(embs, 1) / np.sqrt(n_heads)
#    print('Endpoints')
#    for k,v in endpoints.items():
#        print((k,v))
    return endpoints, None


# uniform bias not used
def make_inceptionv1bn_multi_embeddings(batch_imgs, embedding_dims, n_heads, phase_is_train, uniform_bias=False, weight_decay=0.00004, pooling='avg'):
    # Slim output layer names
    # 'Mixed_3b', 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c'
    emb_info = ['Mixed_3b', 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    if(n_heads==1):
        emb_info = ['Mixed_5c']

    left_embedding = embedding_dims
    with slim.arg_scope(inception_v1.inception_v1_arg_scope(weight_decay=weight_decay)):
        net, endpoints = inception_v1.inception_v1(batch_imgs,
            num_classes=0,
            dropout_keep_prob = 1.0, # output before dropout layer is returned if num_classes is 0
            is_training = phase_is_train
            )
        for i in range(n_heads):
            emb_dim = int( math.ceil(left_embedding / float(n_heads-i) ) ) # put the residual in the preceding embeddings
            left_embedding -= emb_dim
            with tf.variable_scope('loss%d'%i) as scope:
                emb1 = tf.reduce_mean(endpoints[emb_info[i]], [1,2])
                final_emb = emb1
                if(pooling=='avgnmax'):
                    emb2 = tf.reduce_max(endpoints[emb_info[i]], [1,2])
                    final_emb = tf.concat([emb1, emb2], 1)
                endpoints['emb_%d'%i] = slim.fully_connected( final_emb,
                     emb_dim,
                     activation_fn=None)
                endpoints['embedding%d'%i] = tf.nn.l2_normalize( endpoints['emb_%d'%i], dim=1)

        with tf.variable_scope('fc_embedding') as scope:
            embs = [ endpoints['embedding%d'%i] for i in range(n_heads) ]
            endpoints['fc_embedding'] = tf.concat(embs, 1) / np.sqrt(n_heads)

#    print('Endpoints')
#    for k,v in endpoints.items():
#        print((k,v))
    return endpoints, None


###############################################################################
def hdc_one_loss(fc_embedding, label_input, hard_ratio=1.0, negative_margin = 0.5, positive_weight=0.5 ):
    batch_size = tf.shape( fc_embedding ) [0]
    max_cnt = tf.to_float(batch_size*batch_size)
    norm_emb = fc_embedding # alreadd normed. tf.nn.l2_normalize( fc_embedding, 1)
    b_eye = tf.eye( batch_size ) # tf.diag( tf.ones([batch_size]) ) # r0.11 doesn't have tf.eye

    # sims is the cosine similarities of shape [batch_size, batch_size]
    sims = tf.matmul( norm_emb, norm_emb, transpose_b=True, name='sims')

    lbl_ex1 = tf.expand_dims( label_input, 1 , name = 'lbl_ex1')
    lbl1 = tf.tile( lbl_ex1  , [1,batch_size] , name = 'lbl1')
    lbl_ex2 = tf.expand_dims( label_input, 0 , name = 'lbl_ex2')
    lbl2 = tf.tile( lbl_ex2  , [batch_size,1] , name = 'lbl2')

    positive_pair = tf.to_float(tf.equal( lbl1, lbl2 ), name='ispositive') - b_eye # indicators of positive pairs in the batch
    negative_pair = tf.to_float(tf.not_equal( lbl1, lbl2 ), name = 'isnegative') # indicators of negative pairs in the batch

    positive_loss = -1. * positive_pair * sims # 0.5 since positive pairs are duplicated at UR and LL side
    negative_loss = negative_pair * tf.nn.relu(sims - negative_margin)
    pos_normalizer = tf.clip_by_value( tf.reduce_sum(positive_pair), 1., max_cnt )

    positive_loss_sum = tf.reduce_sum( positive_loss ) * positive_weight / pos_normalizer

    if hard_ratio < 1.0 :
        hard_count = tf.clip_by_value( hard_ratio * tf.reduce_sum(negative_pair), 1., max_cnt )
        flat_neg_loss = tf.reshape( negative_loss, [-1] )
        values, indices = tf.nn.top_k( flat_neg_loss, k=tf.to_int32( hard_count ) )

        neg_nz_only = tf.to_float(tf.greater(values, 0. ))
        neg_normalizer = tf.clip_by_value( tf.reduce_sum(neg_nz_only), 1., hard_count )
        negative_loss_hdc = tf.reduce_sum( values )* (1-positive_weight) / neg_normalizer
    else:
        neg_nz_only = tf.to_float(tf.greater(negative_loss, 0. ))
        neg_normalizer = tf.clip_by_value( tf.reduce_sum(neg_nz_only), 1., max_cnt )
        negative_loss_hdc = tf.reduce_sum( negative_loss ) * (1-positive_weight) / neg_normalizer

    loss = tf.add( positive_loss_sum, negative_loss_hdc, name='final_loss')

    return loss


###############################################################################
def hdc_one_loss_sn(fc_embedding, label_input, hard_ratio=1.0, negative_margin = 0.5, positive_weight=0.5 ):
    batch_size = tf.shape( fc_embedding ) [0]
    max_cnt = tf.to_float(batch_size*batch_size)
    norm_emb = fc_embedding # alreadd normed. tf.nn.l2_normalize( fc_embedding, 1)
    b_eye = tf.eye( batch_size ) # tf.diag( tf.ones([batch_size]) ) # r0.11 doesn't have tf.eye

    # sims is the cosine similarities of shape [batch_size, batch_size]
    sims = tf.matmul( norm_emb, norm_emb, transpose_b=True, name='sims')

    lbl_ex1 = tf.expand_dims( label_input, 1 , name = 'lbl_ex1')
    lbl1 = tf.tile( lbl_ex1  , [1,batch_size] , name = 'lbl1')
    lbl_ex2 = tf.expand_dims( label_input, 0 , name = 'lbl_ex2')
    lbl2 = tf.tile( lbl_ex2  , [batch_size,1] , name = 'lbl2')

    positive_pair = tf.to_float(tf.equal( lbl1, lbl2 ), name='ispositive') - b_eye # indicators of positive pairs in the batch
    negative_pair = tf.to_float(tf.not_equal( lbl1, lbl2 ), name = 'isnegative') # indicators of negative pairs in the batch

    positive_loss = -1 * positive_pair * sims # 0.5 since positive pairs are duplicated at UR and LL side
    pos_pair_count = tf.reduce_sum( tf.to_float( tf.greater(positive_pair, 0. )) , 1, name = 'pos_pair_count' )
    pos_pair_only = tf.reduce_sum( positive_loss , 1 ) / tf.clip_by_value( pos_pair_count, 1., max_cnt )
    positive_loss_sum = tf.reduce_mean( pos_pair_only ) * positive_weight

    negative_loss = negative_pair * tf.nn.relu(sims - negative_margin)

    if hard_ratio < 1.0 :
        avg_neg_count = tf.reduce_mean( tf.reduce_sum(negative_pair, 1) )
        hard_count = tf.clip_by_value( tf.ceil( hard_ratio * avg_neg_count ) , 1., max_cnt )
        values, indices = tf.nn.top_k( negative_loss, k=tf.to_int32( hard_count ) )

        neg_nz_only = tf.to_float(tf.greater(values, 0. ))
        neg_normalizer = tf.clip_by_value( tf.reduce_sum(neg_nz_only), 1., max_cnt )
        negative_loss_hdc = tf.reduce_sum( values )* (1-positive_weight) / neg_normalizer
    else:
        neg_nz_count =  tf.reduce_sum( tf.to_float(tf.greater(negative_loss, 0. )) , 1, name = 'neg_nz_count' )
        neg_nz_only = tf.reduce_sum( negative_loss , 1 ) / tf.clip_by_value( neg_nz_count, 1., max_cnt )
        negative_loss_hdc = tf.reduce_mean( neg_nz_only ) *(1.-positive_weight)

    loss = tf.add( positive_loss_sum, negative_loss_hdc, name='final_loss')

    return loss
