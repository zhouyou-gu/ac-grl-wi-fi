import os.path
from os.path import expanduser

from sim_src.ns3_ctrl.ns3_ctrl import build_ns3

home = expanduser("~")
ns3_path = os.path.join(home,"ns-3-dev")
build_ns3(ns3_path)
