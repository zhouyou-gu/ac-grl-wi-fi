import os.path
from os.path import expanduser

from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from working_dir_path import *


build_ns3(get_ns3_path())