import os
import time
from multiprocessing import Process
from os.path import expanduser
from working_dir_path import *

CTRL_PATH = get_controller_path()
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    "test_online_model_40_40_weight_refresh.py",
    "test_online_model_40_40_weight_freeze.py",
    "test_online_model_20_20_weight_refresh.py",
    "test_online_model_20_20_weight_freeze.py",
    "test_online_model_10_10_weight_refresh.py",
    "test_online_model_10_10_weight_freeze.py",
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)


for cmd in CMD_LIST:
    loop = 5
    for ii in range(loop):
        n_test = 8
        p_list = []
        for n in range(n_test):
            seed = n + ii * loop
            tmp_cmd = cmd + ' ' + str(seed)
            print(tmp_cmd)
            p = Process(target=run_cmd, args=(tmp_cmd,))
            p_list.append(p)

        for p in p_list:
            p.start()
            time.sleep(10)

        for p in p_list:
            p.join()
            time.sleep(10)
    time.sleep(10)

