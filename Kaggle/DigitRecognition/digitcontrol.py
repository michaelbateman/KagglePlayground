

# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys
import subprocess

num_samples = int(sys.argv[1])

for my_degree in range(1,10):
	
	call_string = 'python digitfunctions.py ' + str(num_samples) + ' ' + str(my_degree)
	#print call_string
	subprocess.call(call_string, shell=True)	
