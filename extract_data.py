import numpy as np 
import pandas as pd 
from undaqTools import Daq

daq = Daq()
daq.read('PID01_SH_PHD_MainRun_20200124110551.daq')