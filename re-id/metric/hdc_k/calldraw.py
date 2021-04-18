#!/usr/bin/python
import sys
import itertools
import subprocess 
# print the experiments folder
ex_prefix = "ex"
expath2= lambda p: [ ( p+" train", ex_prefix+p+"/save/evalc_train.dmp" ) , ( p+" test", ex_prefix+p+"/save/evalc_test.dmp" ), ( p+" train", ex_prefix+p+"/evalc_train.dmp" ) , ( p+" test", ex_prefix+p+"/evalc_test.dmp" ) ]
methods = list( itertools.chain( *[ expath2(p) for p in sys.argv[1:] ] ))
executable = 'python'
scriptname = 'draweval.py'
args = ['-z', '-c', '2', '-a', '-m', str(methods) ]
subprocess.call([executable, scriptname] + args)
