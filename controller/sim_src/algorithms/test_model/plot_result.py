import os
import time
from multiprocessing import Process
from os.path import expanduser
import matplotlib.pyplot as plt

import numpy as np

CTRL_PATH = os.path.join(expanduser("~"),"wifi-ai/controller")
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = ["test_base_model.py",
             "test_complete_model.py",
             "test_infer_then_label_model.py",
             "test_weight_drift_protection_model.py"]


def load_res_dir_data(dir,cls_name,key,postfix):
    d_name = "%s.%s.%s.txt" % (cls_name,key,postfix)
    d_file = os.path.join(dir,d_name)
    return np.loadtxt(d_file,delimiter=',')

def load_res_dir_parm(dir,cls_name,postfix):
    d_name = "%s.%s.txt" % (cls_name,postfix)
    d_file = os.path.join(dir,d_name)
    return np.genfromtxt(d_file,delimiter=',',dtype=[("name", str, 15),("value",float)]).tolist()

def get_uf_for_alpha(data,a,m_avg = 20):
    res = np.min(data,axis=1)
    if a == 1:
        data = np.log(data+0.001)
    else:
        data = np.power(data+0.001,1-a)/(1-a)
    res = np.convolve(res, np.ones(m_avg)/m_avg, mode='same')
    return res

ALL_ALPHA = [0, 1, 2, 5, 10]
N_STEP = 1000
for t in TEST_LIST:
    DIR = os.path.splitext(t)[0]
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DIR)
    print(DIR_PATH)
    ALL_DIR = [dI for dI in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH,dI))]
    for a in ALL_ALPHA:
        fig = plt.figure(figsize=(16, 6), dpi=80)
        ax = plt.gca()
        print(ax.get_position())
        ax.set_position([0.05, 0.05, 0.675, 0.9])
        FIG_NAME = "%s.ALPHA.%s.jpg"%(DIR,a)
        fig.suptitle(FIG_NAME)

        for dir in ALL_DIR:
            data = load_res_dir_data(os.path.join(DIR_PATH,dir),"sim_env","reward",N_STEP-1)
            print(dir)
            alpha = load_res_dir_parm(os.path.join(DIR_PATH,dir),"sim_config","NaN")
            alpha = int(alpha[1])
            print(alpha)
            if alpha == a:
                res = get_uf_for_alpha(data[:,3:],a)
                plt.plot(data[:,1], res, label=dir)
                plt.plot(data[:,1], res, label=dir)
        plt.legend(bbox_to_anchor=(1.01, 1))
        fig.savefig(os.path.join(DIR_PATH,FIG_NAME))
