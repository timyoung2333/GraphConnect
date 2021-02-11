import os
import csv

def file_proc(file_name):
    res = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row != None:
                for line in row:
                    res.append(eval(line))
    return res

def name_proc(file_name):
    strs = file_name.split('_')
    num = int(strs[1].replace('num', ''))
    wd = float(strs[2].replace('wd', ''))
    lam = float(strs[3].replace('lam', ''))
    bw = float(strs[4].replace('bw', '').replace('.csv', ''))
    return num, wd, lam, bw

def get_all_results(dir_path):
    filenames = os.listdir(dir_path)
    res = {}
    for name in filenames:
        f_dir = dir_path + name
        num, wd, lam, bw = name_proc(name)
        res[num] = {}
        res[num][wd] = {}
        res[num][wd][lam] = {}
        res[num][wd][lam][bw] = file_proc(f_dir)
    return res

dir_path = "./gc_mnist_result/"
res = get_all_results(dir_path)

    


