import os
import time
from multiprocessing import Process
from os.path import expanduser

CTRL_PATH = os.path.join(expanduser("~"),"wifi-ai/controller")
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    # "test_base_model.py",
    # "test_itl_mmf.py",
    "test_online_model_10_10.py",
    "test_online_model_10_40.py",
    # "test_sac_gnn_model.py",
    # "test_reg_gnn_model_bin.py",
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)


for cmd in CMD_LIST:
    for ii in range(11):
        n_test = 10
        p_list = []
        for n in range(n_test):
            print(cmd)
            p = Process(target=run_cmd, args=(cmd,))
            p_list.append(p)

        for p in p_list:
            p.start()
            time.sleep(2)

        for p in p_list:
            p.join()
            time.sleep(2)
    time.sleep(2)
