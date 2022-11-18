import os
from datetime import datetime
import pprint
from time import time


import numpy as np
import torch
def p_true(probability_of_true):
    return np.random.choice([True, False], p=[probability_of_true, 1 - probability_of_true])

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG_TYPE = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    t = torch.from_numpy(ndarray)
    t.requires_grad_(requires_grad)
    if USE_CUDA:
        return t.type(dtype).to(torch.cuda.current_device())
    else:
        return t.type(dtype)
def to_device(var):
    if USE_CUDA:
        return var.to(torch.cuda.current_device())
    return var

def cat_str_dot_txt(sl):
    ret = ""
    for s in sl:
        ret += s
        ret += "."
    ret += "txt"

    return ret


def soft_update_inplace(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update_inplace(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def add_param_noise_inplace(target, std=0.01):
    for target_param in list(target.parameters()):
        d = np.random.randn(1)
        d = d * std
        d = to_tensor(d, requires_grad=False)
        target_param.data.add_(d)

def get_current_time_str():
    return datetime.now().strftime("%Y-%B-%d-%H-%M-%S")

def counted(f):
    def wrapped(self, *args, **kwargs):
        self.N_STEP += 1
        return f(self, *args, **kwargs)

    return wrapped

def timed(f):
    def wrapped(self, *args):
        ts = time()
        result = f(self, *args)
        te = time()
        print('%s func:%r took: %2.4f sec' % (self, f.__name__, te - ts))
        return result

    return wrapped

LOGGED_NP_DATA_HEADER_SIZE = 3


class ParameterConfig(dict):
    CONFIG_NAME = "sim_config"
    def save(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        data_name = "%s.%s.txt" % (self.CONFIG_NAME,postfix)
        data_path = os.path.join(path,data_name)
        with open(data_path, 'w') as f:
            for key, value in self.items():
                f.write('%15s, %s\n' % (key, value))

class StatusObject:
    N_STEP = 0
    DISABLE_ALL_DEBUG = False
    DEBUG_STEP = 100
    DEBUG = False

    MOVING_AVERAGE_TIME_WINDOW = 100

    INIT_MOVING_AVERAGE = False
    INIT_LOGGED_NP_DATA = False

    MOVING_AVERAGE_DICT = {}
    LOGGED_NP_DATA = {}

    def save(self, path: str, postfix: str):
        pass

    def save_np(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        for key in self.LOGGED_NP_DATA:
            data_name = "%s.%s.%s.txt" % (self.__class__.__name__,key,postfix)
            data_path = os.path.join(path,data_name)
            np.savetxt(data_path, self.LOGGED_NP_DATA[key] , delimiter=',')

    def _add_np_log(self, key, float_row_data, g_step=0):
        if not self.INIT_LOGGED_NP_DATA:
            self.LOGGED_NP_DATA = {}
            self.INIT_LOGGED_NP_DATA = True

        float_row_data = np.squeeze(float_row_data)
        assert isinstance(float_row_data, np.ndarray)
        assert float_row_data.ndim == 1
        if not (key in self.LOGGED_NP_DATA):
            self.LOGGED_NP_DATA[key] = np.zeros((0,float_row_data.size+LOGGED_NP_DATA_HEADER_SIZE))
        assert float_row_data.size + LOGGED_NP_DATA_HEADER_SIZE == self.LOGGED_NP_DATA[key].shape[1]
        s_t = np.array([g_step,self.N_STEP,time()])
        data = np.hstack((s_t,float_row_data))
        self.LOGGED_NP_DATA[key] = np.vstack((self.LOGGED_NP_DATA[key],data))

    def status(self):
        if self.DEBUG:
            pprint.pprint(vars(self))

    def _print(self, *args, **kwargs):
        if self.DEBUG and not StatusObject.DISABLE_ALL_DEBUG and (
                self.N_STEP % self.DEBUG_STEP == 0 or self.N_STEP % self.DEBUG_STEP == 1 or self.N_STEP % self.DEBUG_STEP == 2):
            print(("%6d\t" % self.N_STEP) + " ".join(map(str, args)), **kwargs)

    def _printa(self, *args, **kwargs):
        if self.DEBUG and not StatusObject.DISABLE_ALL_DEBUG:
            print(("%6d\t" % self.N_STEP) + ("%10s\t" % self.__class__.__name__) + " ".join(map(str, args)), **kwargs)

    def _moving_average(self, key, new_value):
        if not self.INIT_MOVING_AVERAGE:
            self.MOVING_AVERAGE_DICT = {}
            self.INIT_MOVING_AVERAGE = True

        if not (key in self.MOVING_AVERAGE_DICT):
            self.MOVING_AVERAGE_DICT[key] = 0.

        if key in self.MOVING_AVERAGE_DICT:
            self.MOVING_AVERAGE_DICT[key] = self.MOVING_AVERAGE_DICT[key] * (1.-1./self.MOVING_AVERAGE_TIME_WINDOW) + 1./self.MOVING_AVERAGE_TIME_WINDOW * new_value
            return self.MOVING_AVERAGE_DICT[key]
        else:
            return 0.

    def _debug(self, debug_step=100):
        self.DEBUG = True
        self.DEBUG_STEP = debug_step


def GET_LOG_PATH_FOR_SIM_SCRIPT(sim_script_path):
    OUT_ALL_SIM_FOLDER = os.path.splitext(os.path.basename(sim_script_path))[0]
    OUT_ALL_SIM_FOLDER = os.path.join(os.path.dirname(os.path.realpath(sim_script_path)), OUT_ALL_SIM_FOLDER)
    try:
        os.mkdir(OUT_ALL_SIM_FOLDER)
    except:
        pass
    SIM_NAME_TIME = os.path.splitext(os.path.basename(sim_script_path))[0] + "-" + get_current_time_str() + "-ail"
    OUT_PER_SIM_FOLDER = os.path.join(OUT_ALL_SIM_FOLDER, SIM_NAME_TIME)
    return OUT_PER_SIM_FOLDER

if __name__ == '__main__':
    a = StatusObject()
    # a.DEBUG = True
    b = StatusObject()
    # b.DEBUG = True
    b._debug()

    b._add_np_log("hello", np.ones(6))
    b._moving_average("hello", 1.)

    c = StatusObject()
    print(a.DEBUG,c.DEBUG,b.DEBUG)
    # c.DEBUG = 10
    print(a.DEBUG,b.LOGGED_NP_DATA,c.LOGGED_NP_DATA)
    print(a.DEBUG,b.MOVING_AVERAGE_DICT,c.MOVING_AVERAGE_DICT)
    # cfg = ParameterConfig()
    # cfg["x"] = 5
    # cfg["x"] = 9
    # OUT_FOLDER = os.path.splitext(os.path.basename(__file__))[0] + "-" + get_current_time_str()
    # OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_FOLDER)
    # cfg.save(OUT_FOLDER,"hello")

    # import time
    # from multiprocessing import Process
    #
    #
    # def func():
    #     time.sleep(10000)
    # p1 = Process(target=func, args=())
    # p2 = Process(target=func, args=())
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
