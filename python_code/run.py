import numpy as np
from subprocess import call
from multiprocessing import Pool
import os
import time


num = [500]
seed = [0]
tc = [3]
wd = [0]
lam = [0.0001]
bw = np.logspace(-3, 3, 20)

def run_process(process):
    os.system("python {}".format(process))

processes = []
for n in num:
    for s in seed:
        for c in tc:
            for w in wd:
                for l in lam:
                    for b in bw:
                        process = f"./python_code/gc_mnist_eachclass.py --seed={s} --num={n} --tc={c} --wd={w} --lam={l} --bw={b}"
                        processes.append(process)

eachtime = 2
i = 0
while i + eachtime < len(processes):
    pool = Pool(processes=eachtime)
    pool.map(run_process, processes[i:i+eachtime])
    i += eachtime

if i < len(processes):
    pool = Pool(processes=len(processes)-i)
    pool.map(run_process, processes[i:len(processes)])