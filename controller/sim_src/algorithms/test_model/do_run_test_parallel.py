import os
import time
from multiprocessing import Process
from os.path import expanduser

CTRL_PATH = os.path.join(expanduser("~"),"wifi-ai/controller")
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    # "test_base_model.py",
    # "test_itl_mmf.py",
    # "test_itl_mmf_bidirection_interference.py",
    "test_sac_gnn_model.py",
    "test_reg_gnn_model_bin.py",
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)

for ii in range(5):
    n_test = 3
    p_list = []
    for cmd in CMD_LIST:
        for n in range(n_test):
            print(cmd)
            p = Process(target=run_cmd, args=(cmd,))
            p_list.append(p)

    for p in p_list:
        p.start()
        time.sleep(2)

    for p in p_list:
        p.join()
