import os
from multiprocessing import Process
from os.path import expanduser

CTRL_PATH = os.path.join(expanduser("~"),"wifi-ai/controller")
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = ["test_base_model.py",
             "test_complete_model.py",
             "test_infer_then_label_model.py",
             "test_weight_drift_protection_model.py"]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)


p_list = []
for cmd in CMD_LIST:
    print(cmd)
    p = Process(target=run_cmd, args=(cmd,))
    p_list.append(p)

for p in p_list:
    p.start()

for p in p_list:
    p.join()
