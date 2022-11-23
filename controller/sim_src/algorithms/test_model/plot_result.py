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
             "test_weight_drift_protection_model.py",
             "test_min_fair_model.py",
             "test_itl_mmf.py"
             "test_dac_gnn_model.py",
             "test_sac_gnn_model.py",
             ]


def load_res_dir_data(dir,cls_name,key,postfix):
    d_name = "%s.%s.%s.txt" % (cls_name,key,postfix)
    d_file = os.path.join(dir,d_name)
    if os.path.isfile(d_file):
        return np.loadtxt(d_file,delimiter=',')

    d_name = "%s_dac_gnn.%s.%s.txt" % (cls_name,key,postfix)
    d_file = os.path.join(dir,d_name)
    if os.path.isfile(d_file):
        return np.loadtxt(d_file,delimiter=',')

    d_name = "%s_sac_gnn.%s.%s.txt" % (cls_name,key,postfix)
    d_file = os.path.join(dir,d_name)
    if os.path.isfile(d_file):
        return np.loadtxt(d_file,delimiter=',')

def load_res_dir_parm(dir,cls_name,postfix):
    d_name = "%s.%s.txt" % (cls_name,postfix)
    d_file = os.path.join(dir,d_name)
    return np.genfromtxt(d_file,delimiter=',',dtype=[("name", str, 15),("value",float)]).tolist()

def get_uf_value(data,a,m_avg = 50):
    if a == 1:
        data = np.log(data+0.001)
        data = np.mean(data,axis=1)
        res = np.exp(data)
    else:
        data = np.power(data+0.001,1-a)
        data = np.mean(data,axis=1)
        res = np.power(data,1./(1.-a))
    res = np.convolve(res, np.ones(m_avg)/m_avg, mode='same')
    return res

def get_min_rate(data,a,m_avg = 100):
    res = np.min(data,axis=1)
    res = np.convolve(res, np.ones(m_avg)/m_avg, mode='same')
    return res


def get_op(data,a,m_avg = 50):
    data[data>0.05] = 1.
    data = 1-data
    res = np.mean(data,axis=1)
    res = np.convolve(res, np.ones(m_avg)/m_avg, mode='same')
    return res

def get_mean_rate(data,a,m_avg = 50):
    res = np.mean(data,axis=1)
    res = np.convolve(res, np.ones(m_avg)/m_avg, mode='same')
    return res

ALL_ALPHA = [10,100]
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
        ax.set_position([0.05, 0.05, 0.55, 0.9])
        FIG_NAME = "%s.ALPHA.%s.uf_value.jpg"%(DIR,a)
        fig.suptitle(FIG_NAME)
        ax.set_xlim([0, N_STEP])
        ax.set_ylim([0, 1])
        for dir in ALL_DIR:
            print(dir)
            alpha = load_res_dir_parm(os.path.join(DIR_PATH,dir),"sim_config","NaN")
            alpha = int(alpha[1])
            print(alpha)
            if alpha == a:
                data = load_res_dir_data(os.path.join(DIR_PATH,dir),"sim_env","reward",N_STEP-1)
                res = get_uf_value(data[:,3:],a)
                plt.plot(data[:,1], res, label=dir)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.savefig(os.path.join(DIR_PATH,FIG_NAME))

for t in TEST_LIST:
    DIR = os.path.splitext(t)[0]
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DIR)
    print(DIR_PATH)
    ALL_DIR = [dI for dI in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH,dI))]
    for a in ALL_ALPHA:
        fig = plt.figure(figsize=(16, 6), dpi=80)
        ax = plt.gca()
        print(ax.get_position())
        ax.set_position([0.05, 0.05, 0.55, 0.9])
        FIG_NAME = "%s.ALPHA.%s.min_rate.jpg"%(DIR,a)
        fig.suptitle(FIG_NAME)
        ax.set_xlim([0, N_STEP])
        ax.set_ylim([0, 0.2])
        for dir in ALL_DIR:
            print(dir)
            alpha = load_res_dir_parm(os.path.join(DIR_PATH,dir),"sim_config","NaN")
            alpha = int(alpha[1])
            print(alpha)
            if alpha == a:
                data = load_res_dir_data(os.path.join(DIR_PATH,dir),"sim_env","reward",N_STEP-1)
                res = get_min_rate(data[:,3:],a)
                plt.plot(data[:,1], res, label=dir)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.savefig(os.path.join(DIR_PATH,FIG_NAME))

for t in TEST_LIST:
    DIR = os.path.splitext(t)[0]
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DIR)
    print(DIR_PATH)
    ALL_DIR = [dI for dI in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH,dI))]
    for a in ALL_ALPHA:
        fig = plt.figure(figsize=(16, 6), dpi=80)
        ax = plt.gca()
        print(ax.get_position())
        ax.set_position([0.05, 0.05, 0.55, 0.9])
        FIG_NAME = "%s.ALPHA.%s.outage_prob.jpg"%(DIR,a)
        fig.suptitle(FIG_NAME)
        ax.set_xlim([0, N_STEP])
        ax.set_ylim([0, 0.02])
        for dir in ALL_DIR:
            print(dir)
            alpha = load_res_dir_parm(os.path.join(DIR_PATH,dir),"sim_config","NaN")
            alpha = int(alpha[1])
            print(alpha)
            if alpha == a:
                data = load_res_dir_data(os.path.join(DIR_PATH,dir),"sim_env","reward",N_STEP-1)
                res = get_op(data[:,3:],a)
                plt.plot(data[:,1], res, label=dir)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.savefig(os.path.join(DIR_PATH,FIG_NAME))

for t in TEST_LIST:
    DIR = os.path.splitext(t)[0]
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DIR)
    print(DIR_PATH)
    ALL_DIR = [dI for dI in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH,dI))]
    for a in ALL_ALPHA:
        fig = plt.figure(figsize=(16, 6), dpi=80)
        ax = plt.gca()
        print(ax.get_position())
        ax.set_position([0.05, 0.05, 0.55, 0.9])
        FIG_NAME = "%s.ALPHA.%s.sum_rate.jpg"%(DIR,a)
        fig.suptitle(FIG_NAME)
        ax.set_xlim([0, N_STEP])
        ax.set_ylim([0, 0.5])
        for dir in ALL_DIR:
            print(dir)
            alpha = load_res_dir_parm(os.path.join(DIR_PATH,dir),"sim_config","NaN")
            alpha = int(alpha[1])
            print(alpha)
            if alpha == a:
                data = load_res_dir_data(os.path.join(DIR_PATH,dir),"sim_env","reward",N_STEP-1)
                res = get_mean_rate(data[:,3:],a)
                plt.plot(data[:,1], res, label=dir)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.savefig(os.path.join(DIR_PATH,FIG_NAME))