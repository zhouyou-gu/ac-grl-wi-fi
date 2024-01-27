import os
from os.path import expanduser


# add your path here as: default_path = "somepath/ac-grl-wi-fi";
# otherwise, the codes assume the path is "~/ac-grl-wi-fi"
default_path = None


def get_working_dir_path():
    if default_path == None:
        home = expanduser("~")
        path = os.path.join(home,"ac-grl-wi-fi")
        return path
    else:
        return default_path
def get_ns3_path():
    path =  get_working_dir_path()
    return os.path.join(path,"ns-3-dev")

def get_controller_path():
    path =  get_working_dir_path()
    return os.path.join(path,"controller")