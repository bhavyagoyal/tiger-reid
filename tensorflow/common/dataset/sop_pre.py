"""
This is Stanford Online DataSet preprocessing based on the author's github
 https://github.com/rksltnl/Deep-Metric-Learning-CVPR16
 The preprocesing code is "https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/blob/master/code/load_cropped_images.m"
 It crop the white border around the image.
 The default configuration came from "https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/blob/master/code/config.m"

Author: keunjoo.kwon@samsung.com
Copyright (c) 2017 Samsung Electronics Co., Ltd. All Rights Reserved
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
import glob
import itertools
import traceback
import signal
import multiprocessing as mp
import argparse

# pylint: disable=line-too-long
###############################################################################
def pil_crop_white_border(im_org, crop_padding):
    im_gray = im_org.convert("L") 

    data = np.array( im_gray ).astype( np.uint8, copy=False )
    rows, cols = np.where( data < 250 )

    xmin = max(0, np.min(cols) - crop_padding)
    xmax = min(data.shape[1], max(cols)+1 + crop_padding)
    ymin = max(0, np.min(rows) - crop_padding)
    ymax = min(data.shape[0], max(rows)+1 + crop_padding)

    im_cropped = im_org.crop([xmin, ymin, xmax, ymax])

    return im_cropped 

def pil_square_scale(im, square_size, bgColor = '#FFFFFF' ): 
    if im.size[0] == im.size[1]:
        nim = im
    else:
        nim = PIL.Image.new( im.mode, ( max(im.size),max(im.size) ), bgColor )
        x_offset = (max(im.size) - im.size[0]) //2
        y_offset = (max(im.size) - im.size[1]) //2
        nim.paste( im, (x_offset, y_offset) )

    nim2 = nim.resize( (square_size, square_size) , PIL.Image.BICUBIC)
    return nim2

###############################################################################
def input_generator(args):
    for fname in os.listdir( args.input_dir ):
        input_file = os.path.join( args.input_dir, fname )
        yield input_file, args.output_dir, args.crop_padding, args.force_square_size

def convert_one_image(params):
    input_file, output_dir, crop_padding, force_square_size = params
    try:
        im = PIL.Image.open(input_file)
        im = im.convert("RGB") if im.mode != "RGB" else im
        im_cropped = pil_crop_white_border(im, crop_padding)
        im_final = pil_square_scale(im_cropped, force_square_size ) if force_square_size > 0 else im_cropped
        fname = os.path.basename(input_file)
        im_final.save( os.path.join( output_dir, fname ) )
    except Exception as ex:
        return ex, input_file
        
    return None, input_file

def convert_all(args):
    pool = mp.Pool(args.num_processes, signal.signal, (signal.SIGINT, signal.SIG_IGN) )
    try:
        params = input_generator(args)
        convert_result = pool.imap_unordered( convert_one_image, params )
        #-----------------------------------------------------------------
        for i in itertools.count():
            ex, input_file = convert_result.next(timeout=sys.maxint)
            if ex is not None:
                print(i, input_file, ex)
            if i%100 == 0:
                sys.stderr.write('.')
                if i%1000 == 0:
                    sys.stderr.write('%d\n'%i)
        #-----------------------------------------------------------------
    except StopIteration as ex:
        print ("Done.", file=sys.stderr)
    except KeyboardInterrupt as ex:
        print ( type(ex), file=sys.stderr)
    except Exception as ex:
        traceback.print_exc()
    finally:
        print("Joining the pool", file=sys.stderr)
        pool.close()
        pool.terminate()
        pool.join()

###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='the path of the input folder')
    parser.add_argument('-o','--output_dir', help='output folder', required =True)
    parser.add_argument('-c','--crop_padding', help='pixels to pad around the image, default=15', type=int, default=15 )
    parser.add_argument('-f','--force_square_size', help='make the image square if this option is positive, default=0', type=int, default=0 )
    parser.add_argument('-j','--num_processes', help='# of processes, default=1', type=int, default=1 )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print('making', args.output_dir)
        os.mkdir( args.output_dir ) 

    convert_all(args)

if __name__ == "__main__":
    main()
