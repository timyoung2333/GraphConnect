import os
import csv
import numpy as np

def file_proc(file_name, num_epochs):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        res = [eval(line) for row in reader for line in row]
        if len(res) == 0:
            res = np.zeros((num_epochs, 4))
            # print(file_name)
    return np.array(res)

def name_proc(file_name):
    strs = file_name.split('_')
    num = int(strs[1].replace('num', ''))
    wd = float(strs[2].replace('wd', ''))
    lam = float(strs[3].replace('lam', ''))
    bw = float(strs[4].replace('bw', '').replace('.csv', ''))
    return num, wd, lam, bw

def get_all_results(dir_path, num_epochs):
    filenames = os.listdir(dir_path)
    for name in filenames:
        f_dir = dir_path + name
        res = file_proc(f_dir, num_epochs)

    num, wd, lam, bw = [], [], [], []
    for name in filenames:
        n, w, l, b = name_proc(name)
        if n not in num:
            num.append(n)
        if w not in wd:
            wd.append(w)
        if l not in lam:
            lam.append(l)
        if b not in bw:
            bw.append(b)
    num.sort()
    wd.sort()
    lam.sort()
    bw.sort()

    # print(f'num = {num}\nwd = {wd}\nlam = {lam}\nbw = {bw}')

    d_num = dict(zip(num, range(len(num))))
    d_wd = dict(zip(wd, range(len(wd))))
    d_lam = dict(zip(lam, range(len(lam))))
    d_bw = dict(zip(bw, range(len(bw))))

    dicts = [d_num, d_wd, d_lam, d_bw]

    l = [len(num), len(wd), len(lam), len(bw)]
    res = np.zeros((*l, num_epochs, 4))
    for name in filenames:
        f_dir = dir_path + name
        num, wd, lam, bw = name_proc(name)
        res[d_num[num], d_wd[wd], d_lam[lam], d_bw[bw], :] = file_proc(f_dir, num_epochs)
    return res, dicts

cur_dir = os.getcwd()
if 'python_code' in cur_dir:
    cur_dir = cur_dir.replace('/python_code', '')
res_dir = cur_dir + '/gc_mnist_result/'
res, dicts = get_all_results(res_dir, 200)


