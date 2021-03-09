import os
from file_proc_eachclass import get_all_results
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use("Qt5Agg")




def plt_bw(test_class):

    cur_dir = os.getcwd()
    if 'python_code' in cur_dir:
        cur_dir = cur_dir.replace('/python_code', '')
    res_dir = cur_dir + '/gc_mnist_result_eachclass/'
    res, dicts = get_all_results(res_dir, 150, test_class)
    num_dict, wd_dict, lam_dict, bw_dict = dicts[0], dicts[1], dicts[2], dicts[3]

    ## first few figures, x-axis bandwidth
    # bw vs loss & acc
    x = list(bw_dict.keys())
    lams = [0.0001]
    num = 200
    fig = plt.figure()

    plt.subplot(131)
    for lam in lams:
        y = res[num_dict[num], wd_dict[0], lam_dict[lam], :, -1, :]
        xx = np.array([x[i] for i in range(len(x)) if y[i, 1] != 0])
        yy = np.array([y[i] for i in range(len(x)) if y[i, 1] != 0])
        plt.plot(xx, yy[:, 1], ls='-', marker='.', label=f'train loss, lam={lam}')
    plt.xscale('log')
    plt.xlabel('bandwidth')
    plt.ylabel('training loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(132)
    for lam in lams:
        y = res[num_dict[num], wd_dict[0], lam_dict[lam], :, -1, :]
        xx = np.array([x[i] for i in range(len(x)) if y[i, 2] != 0])
        yy = np.array([y[i] for i in range(len(x)) if y[i, 2] != 0])
        plt.plot(xx, yy[:, 2], ls='-', marker='.', label=f'test loss, lam={lam}')
    plt.xscale('log')
    plt.xlabel('bandwidth coefficient')
    plt.ylabel('test loss')
    plt.legend()
    plt.grid(True)


    plt.subplot(133)
    for lam in lams:
        y = res[num_dict[num], wd_dict[0], lam_dict[lam], :, -1, :]
        xx = np.array([x[i] for i in range(len(x)) if y[i, 3] != 0])
        yy = np.array([y[i] for i in range(len(x)) if y[i, 3] != 0])
        plt.plot(xx, yy[:, 3], ls='-', marker='.', label=f'accuracy, lam={lam}')
    plt.xscale('log')
    plt.xlabel('bandwidth')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    # mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(50,100,2000, 545)
    fig.savefig(f"{test_class}.pdf")
    plt.show()

for i in range(10):
    plt_bw(i)

# ## second few figures
# # lambda vs. loss
# x = list(lam_dict.keys())
# # x = np.array([x[i] in x for i in range(len(x)) if x[i] < 0.1])
# bws = list(bw_dict.keys())
# bws = [bws[2 * i] for i in range(len(bws)//2)]
# num = 500
# fig = plt.figure(2)

# plt.subplot(131)
# for bw in bws:
#     y = res[num_dict[num], wd_dict[0],:, bw_dict[bw], -1, :]
#     xx = np.array([x[i] for i in range(len(x)) if y[i, 1] != 0])
#     yy = np.array([y[i] for i in range(len(x)) if y[i, 1] != 0])
#     plt.plot(xx, yy[:, 1], ls='-', marker='.', label=f'train loss, bw={bw}')
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('training loss')
# plt.legend()
# plt.grid(True)

# plt.subplot(132)
# for bw in bws:
#     y = res[num_dict[num], wd_dict[0],:, bw_dict[bw], -1, :]
#     xx = np.array([x[i] for i in range(len(x)) if y[i, 2] != 0])
#     yy = np.array([y[i] for i in range(len(x)) if y[i, 2] != 0])
#     plt.plot(xx, yy[:, 2], ls='-', marker='.', label=f'test loss, bw={bw}')
#     plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('test loss')
# plt.legend()
# plt.grid(True)


# plt.subplot(133)
# for bw in bws:
#     y = res[num_dict[num], wd_dict[0],:, bw_dict[bw], -1, :]
#     xx = np.array([x[i] for i in range(len(x)) if y[i, 3] != 0])
#     yy = np.array([y[i] for i in range(len(x)) if y[i, 3] != 0])
#     plt.plot(xx, yy[:, 3], ls='-', marker='.', label=f'accuracy, bw={bw}')
#     plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('accuracy')
# plt.legend()
# plt.grid(True)
# mngr = plt.get_current_fig_manager()
# mngr.window.setGeometry(50,100,2000, 545)

# plt.show()

# ## third few figures
# # num vs. epochs
# x = list(range(200))
# # x = np.array([x[i] in x for i in range(len(x)) if x[i] < 0.1])
# nums = [100, 200, 500, 1000]

# fig = plt.figure(3)

# plt.subplot(121)
# for num in nums:
#     y = res[num_dict[num], wd_dict[0], lam_dict[0.0001], bw_dict[12.7427], :, :]
#     plt.plot(x, y[:, 1], ls='-', marker=None, label=f'train loss, num={num}')
#     plt.plot(x, y[:, 2], ls='--', marker=None, label=f'test loss, num={num}')
# plt.xlabel('epochs')
# plt.ylabel('training loss')
# plt.legend()
# plt.grid(True)



# plt.subplot(122)
# for num in nums:
#     y = res[num_dict[num], wd_dict[0], lam_dict[0.0001], bw_dict[12.7427], :, :]
#     plt.plot(x, y[:, 3], ls='-', marker=None, label=f'accuracy, num={num}')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.grid(True)

# mngr = plt.get_current_fig_manager()
# mngr.window.setGeometry(50,100,1500, 545)

# plt.show()


