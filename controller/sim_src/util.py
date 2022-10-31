import os
from datetime import datetime
import pprint
from time import time


import numpy as np
import torch

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
    DEBUG_STEP = 1000
    DEBUG = False

    MOVING_AVERAGE_TIME_WINDOW = 100
    MOVING_AVERAGE_DICT = {}
    LOGGED_NP_DATA = {}

    def save(self, path: str, postfix: str):
        pass
    def _add_np_log_field(self, key:str, length:int):
        self.LOGGED_NP_DATA[key] = np.zeros((0,length+LOGGED_NP_DATA_HEADER_SIZE))

    def save_np(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        for key in self.LOGGED_NP_DATA:
            data_name = "%s.%s.%s.txt" % (self.__class__.__name__,key,postfix)
            data_path = os.path.join(path,data_name)
            np.savetxt(data_path, self.LOGGED_NP_DATA[key] , delimiter=',')

    def add_np_log(self, key, float_row_data, g_step=0):
        float_row_data = np.squeeze(float_row_data)
        assert isinstance(float_row_data, np.ndarray)
        assert float_row_data.ndim == 1
        if not (key in self.LOGGED_NP_DATA):
            self._add_np_log_field(key,float_row_data.size)
        assert float_row_data.size + LOGGED_NP_DATA_HEADER_SIZE == self.LOGGED_NP_DATA[key].shape[1]
        s_t = np.array([g_step,self.N_STEP,time()])
        data = np.hstack((s_t,float_row_data))
        self.LOGGED_NP_DATA[key] = np.vstack((self.LOGGED_NP_DATA[key],data))

    def status(self):
        if self.DEBUG:
            pprint.pprint(vars(self))

    def _print(self, *args, **kwargs):
        if self.DEBUG and (
                self.N_STEP % self.DEBUG_STEP == 0 or self.N_STEP % self.DEBUG_STEP == 1 or self.N_STEP % self.DEBUG_STEP == 2):
            print(("%6d\t" % self.N_STEP) + " ".join(map(str, args)), **kwargs)
    def _printa(self, *args, **kwargs):
        if self.DEBUG:
            print(("%6d\t" % self.N_STEP) + ("%10s\t" % self.__class__.__name__) + " ".join(map(str, args)), **kwargs)
    def _moving_average(self, key, new_value):
        if key in self.MOVING_AVERAGE_DICT:
            self.MOVING_AVERAGE_DICT[key] = self.MOVING_AVERAGE_DICT[key] * (1.-1./self.MOVING_AVERAGE_TIME_WINDOW) + 1./self.MOVING_AVERAGE_TIME_WINDOW * new_value
            return self.MOVING_AVERAGE_DICT[key]
        else:
            return 0.

if __name__ == '__main__':
    a = StatusObject()
    a.DEBUG = True
    b = StatusObject()
    c = StatusObject()
    c.DEBUG = 10
    print(a.DEBUG,b.DEBUG,c.DEBUG)
    cfg = ParameterConfig()
    cfg["x"] = 5
    cfg["x"] = 9
    OUT_FOLDER = os.path.splitext(os.path.basename(__file__))[0] + "-" + get_current_time_str()
    OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_FOLDER)
    cfg.save(OUT_FOLDER,"hello")
