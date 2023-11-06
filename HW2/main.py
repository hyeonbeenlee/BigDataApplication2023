import numpy as np
import pandas as pd
import glob
import os
from helpers import *


assert (
    os.getcwd().split("/")[-1] == "HW2"
), "You're working at the wrong directory. Reopen VScode in the folder 'HW2'."

# convert_to_csv()

series_list, df = read_describe()

df_fix = cleansing(df)

df_fix = outliers_quantile(df_fix)
# df_fix = outliers_zscore(df)

df_fix_log = log_transformation(df_fix)

normality_test(df_fix)
normality_test(df_fix_log)

pearson = pearson_corr(stdize(df))
spearman = spearman_corr(stdize(df))


visualize(df, df_fix)
