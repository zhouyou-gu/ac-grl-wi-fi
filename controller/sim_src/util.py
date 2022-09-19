from datetime import datetime
from time import time


def cat_str_dot_txt(sl):
    ret = ""
    for s in sl:
        ret += s
        ret += "."
    ret += "txt"

    return ret
