import numpy as np
import pandas as pd
import glob

for path in glob.glob('Data/*.xlsx'):
    data=pd.read_excel(path, header=None)
    print(path)
    print(data.describe())
    print('='*100)