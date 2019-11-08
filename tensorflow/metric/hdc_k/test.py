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

import common.dataset.sop as sop


sop.test_batch(True, '/home/bhavya/datasets/CUB/CUB_200_2011/metric_train.txt')
