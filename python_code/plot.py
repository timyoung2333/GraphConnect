import os
from file_proc import get_all_results
import numpy as np
from matplotlib import pyplot as plt

cur_dir = os.getcwd()
if 'python_code' in cur_dir:
    cur_dir = cur_dir.replace('/python_code', '')
res_dir = cur_dir + '/gc_mnist_result/'
res, dicts = get_all_results(res_dir, 200)
num_dict, wd_dict, lam_dict, bw_dict = dicts[0], dicts[1], dicts[2], dicts[3]

## Figure 1:
# let's plot num = 500, 1000, wd = 0, different lams on different bw
# bw vs loss & acc
fig = plt.figure(1)
x = list(bw_dict.keys())
lams = [0.01, 0.001, 0.0001, 1e-5, 0]
for lam in lams:
    y = res[num_dict[500], wd_dict[0], lam_dict[lam], :, -1, :]
    plt.plot(x, y[:, 1], ls='-', marker='.', label=f'train loss, lam={lam}')
plt.xscale('log')
plt.xlabel('bandwidth')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

fig = plt.figure(2)
x = list(bw_dict.keys())
lams = [0.01, 0.001, 0.0001, 1e-5, 0]
for lam in lams:
    y = res[num_dict[500], wd_dict[0], lam_dict[lam], :, -1, :]
    plt.plot(x, y[:, 2], ls='-', marker='.', label=f'test loss, lam={lam}')
plt.xscale('log')
plt.xlabel('bandwidth')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(3)
x = list(bw_dict.keys())
lams = [0.01, 0.001, 0.0001, 1e-5, 0]
for lam in lams:
    y = res[num_dict[500], wd_dict[0], lam_dict[lam], :, -1, :]
    plt.plot(x, y[:, 3], ls='-', marker='.', label=f'accuracy, lam={lam}')
plt.xscale('log')
plt.xlabel('bandwidth')
plt.ylabel('acc')
plt.legend()
plt.grid(True)
plt.show()



# ## Figure 2:
# # lambda vs. loss
# fig = plt.figure(3)
# x = list(lam_dict.keys())
# y = res[num_dict[500], wd_dict[0], :, bw_dict[6.1585], -1, :]
# plt.plot(x, y[:, 1])
# plt.plot(x, y[:, 2])
# plt.xscale('log')
# plt.show()

## Figure 2: 
# epochs vs loss & acc (diff num, lam, bw)


