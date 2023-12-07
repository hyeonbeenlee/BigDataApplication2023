import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.getcwd())
from HW3.helpers import *

assert (
    os.getcwd().split("/")[-1] == "BigDataApplication"
), "Please change directory to BigDataApplication"

series_list, df = read_describe()
df_hourly = df.resample('1H', on='TimeStamp').mean()
pass