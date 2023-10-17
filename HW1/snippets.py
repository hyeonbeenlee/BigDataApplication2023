import pandas as pd
import numpy as np


def check_unique_values(df):
    print(f"{'='*100}")
    for c in df.columns:
        u = pd.unique(df[c])
        if not u.dtype == np.number:
            print(f'"{c}": {u}')
        else:
            print(f'"{c}": NUMERIC VALUES')
        print()
    print(f"{'='*100}")
