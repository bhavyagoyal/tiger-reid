import subprocess
from subprocess import PIPE, STDOUT
import shutil, os
import time
arr1 = [0.1, 0.5, 0.9, 1.0, 5]  
arr2 = [0.5, 0.8, 0.9, 0.95] #ex73s decay rates 
arr3 = [5000, 10000, 15000] #ex73s decay steps 

# Script used for hyperparameter tuning

BASE_FOLDER = 'exT7'
for i,arr1ele in enumerate(arr1):
#    for j,arr2ele in enumerate(arr2):
#        for k,arr3ele in enumerate(arr3):
			NEW_FOLDER = BASE_FOLDER + '_' + chr(97+i) #+ chr(97+j) + chr(97+k)
			if not os.path.exists(NEW_FOLDER):
				os.makedirs(NEW_FOLDER)

			shutil.copy(os.path.join(BASE_FOLDER, 'sub'), NEW_FOLDER)
			lines = open(os.path.join(NEW_FOLDER, 'sub'),'r').readlines()
			lines[22] = lines[22].replace('=0.9 ', '='+str(arr1ele)+' ')
###			lines[30] = lines[30].replace('TRAIN', 'VAL')
###			lines[1] = lines[1].replace('TRAIN', 'VAL')
			f = open(os.path.join(NEW_FOLDER, 'sub'),'w')
			f.write(''.join(lines).replace('\r',''))
			f.close()

       		
#			lines = open(os.path.join(NEW_FOLDER, 'sub'),'r').readlines()
#			p = subprocess.Popen(['bsub'], cwd=NEW_FOLDER, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
#			pstdout = p.communicate(''.join(lines))[0]
#			print pstdout.decode()



