import numpy as np
from subprocess import call
from multiprocessing import Pool
import os
import time


num = [200]
seed = [0]
wd = [0]
lam = [0.0001]
coef = np.arange(0.90, 1, 0.1)

def run_process(process):
    os.system("python {}".format(process))

processes = []
for n in num:
    for s in seed:
            for w in wd:
                for l in lam:
                    for co in coef:
                        process = f"./python_code/gc_mnist_v2.py --seed={s} --num={n} --wd={w} --lam={l} --coef={co}"
                        processes.append(process)

eachtime = 1
i = 0
while i + eachtime < len(processes):
    pool = Pool(processes=eachtime)
    pool.map(run_process, processes[i:i+eachtime])
    i += eachtime

if i < len(processes):
    pool = Pool(processes=len(processes)-i)
    pool.map(run_process, processes[i:len(processes)])