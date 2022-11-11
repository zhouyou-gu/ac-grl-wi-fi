import os
import random
from os.path import expanduser

from sim_src.sim_env.sim_env import sim_env
from sim_src.util import ParameterConfig, StatusObject


def run_test(a,mc,log_path):
    StatusObject.DISABLE_ALL_DEBUG = True
    ALPHA = a
    OUT_FOLDER = log_path
    model_class = mc
    ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
    e = sim_env(id=random.randint(40,200))
    e.PROG_PATH = ns3_path
    e.PROG_NAME = "wifi-ai/env"
    e.DEBUG = True

    n_step = 1000
    batch_size = 1
    model = model_class(0)
    model.DEBUG_STEP = 10
    model.DEBUG = True

    cfg = ParameterConfig()
    cfg['ALPHA'] = ALPHA
    model.FAIRNESS_ALPHA = cfg['ALPHA']
    cfg.save(OUT_FOLDER,"NaN")

    e.set_actor(model)
    for i in range(n_step):
        batch = []
        for j in range(batch_size):
            e.init_env()
            sample = e.step(no_run=False)
            batch.append(sample)
        model.step(batch)
        if (i+1) % 100 == 0:
            model.save(OUT_FOLDER,str(i))
        if (i+1) in [10, 100, 500, 1000]:
            e.save_np(OUT_FOLDER,str(i))