import os
import csv
import numpy as np

def file_proc(file_name, num_epochs, test_class):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        res = []
        for row in reader:
            for line in row:
                l = eval(line)
                res.append([l[0], l[1], l[2][test_class], l[3][test_class]])
        # res = [eval(line) for row in reader for line in row]
        if len(res) == 0:
            res = np.zeros((num_epochs, 4))
            # print(file_name)
    return np.array(res)

def name_proc(file_name):
    strs = file_name.split('_')
    num = int(strs[1].replace('num', ''))
    wd = float(strs[2].replace('wd', ''))
    lam = float(strs[3].replace('lam', ''))
    coef = float(strs[4].replace('coef', '').replace('.csv', ''))
    return num, wd, lam, coef

def get_all_results(dir_path, num_epochs, test_class):
    filenames = os.listdir(dir_path)
    for name in filenames:
        f_dir = dir_path + name
        res = file_proc(f_dir, num_epochs, test_class)

    num, wd, lam, coef = [], [], [], []
    for name in filenames:
        n, w, l, co = name_proc(name)
        if n not in num:
            num.append(n)
        if w not in wd:
            wd.append(w)
        if l not in lam:
            lam.append(l)
        if co not in coef:
            coef.append(co)
    num.sort()
    wd.sort()
    lam.sort()
    coef.sort()

    # print(f'num = {num}\nwd = {wd}\nlam = {lam}\nbw = {bw}')

    d_num = dict(zip(num, range(len(num))))
    d_wd = dict(zip(wd, range(len(wd))))
    d_lam = dict(zip(lam, range(len(lam))))
    d_coef = dict(zip(coef, range(len(coef))))

    dicts = [d_num, d_wd, d_lam, d_coef]

    l = [len(num), len(wd), len(lam), len(coef)]
    res = np.zeros((*l, num_epochs, 4))
    for name in filenames:
        f_dir = dir_path + name
        n, w, l, co = name_proc(name)
        res[d_num[n], d_wd[w], d_lam[l], d_coef[co], :] = file_proc(f_dir, num_epochs, test_class)
    return res, dicts

cur_dir = os.getcwd()
if 'python_code' in cur_dir:
    cur_dir = cur_dir.replace('/python_code', '')
res_dir = cur_dir + '/gc_mnist_result_eachclass/'
res, dicts = get_all_results(res_dir, 150, 2)


