import os
import time
from multiprocessing import Process
from os.path import expanduser

CTRL_PATH = os.path.join(expanduser("~"),"wifi-ai/controller")
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    "test_trained_nn_k20_z2.py",
    # "test_trained_nn_k20_z8.py",
    # "test_trained_nn_k20_z16.py",
    # "test_trained_nn_k20_z32.py",
    # "test_trained_nn_k40_z4.py",
    # "test_trained_nn_k40_z8.py",
    # "test_trained_nn_k40_z16.py",
    # "test_trained_nn_k40_z32.py",
    # "test_trained_nn_k40_z64.py",
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

