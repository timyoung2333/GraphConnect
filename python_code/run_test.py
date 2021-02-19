from subprocess import call
from multiprocessing import Pool
import os
import time

def run_process(process):                                                             
    os.system('python {}'.format(process))   

processes = []
for i in range(60, 80):
    start = time.time()
    processes.append('./python_code/subprocess_test.py')
    pool = Pool(processes=i+1)                                                        
    pool.map(run_process, processes)                       
    print(time.time() - start)