import os
import time
from multiprocessing import Process
from os.path import expanduser
from working_dir_path import *

CTRL_PATH = get_controller_path()
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    "test_bianchi_model_global.py",
    "test_graph_cut_max_contention.py",
    "test_graph_cut_max_hidden.py",
    "test_graph_cut_max_interference.py",
    "test_trained_nn.py",
    "test_twt_in_turn_global.py",
    "test_twt_per_ap_group.py",
    "test_twt_rnd.py",
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)

for ii in range(1):
    n_test = 1
    p_list = []
    for cmd in CMD_LIST:
        for n in range(n_test):
            print(cmd)
            p = Process(target=run_cmd, args=(cmd,))
            p_list.append(p)

    for p in p_list:
        p.start()
        time.sleep(10)

    for p in p_list:
        p.join()
        time.sleep(10)

