import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from HW2.helpers import *
from HW2.helpers import _get_filename


def read_describe_with_timestamp():
    series_list = []
    for idx, path in enumerate(glob.glob("HW2/Data/*.csv")):
        if "TimeStamp" in path:
            datetime = pd.read_csv(
                path,
                names=["year", "month", "day", "hour", "minute"],
                skiprows=[0],
            ).reset_index(drop=True)
            datetime = pd.to_datetime(datetime)
        else:
            data = pd.read_csv(
                path, names=[_get_filename(path)], skiprows=[0]
            ).reset_index(
                drop=True
            )  # series
            print(_get_filename(path))
            series_list.append(data)
        # print(path)
        # print(data.describe())
        # print("=" * 100)
    df = pd.concat(series_list, axis=1)
    df["TimeStamp"] = datetime
    df = df.reindex(sorted(df.columns), axis=1)
    return series_list, df


def HW2_aggregated_run():
    # read
    series_list, df = read_describe()

    # cleanse
    df_fix = cleansing(df)
    
    # log transform
    df_fix = outliers_quantile(df_fix)
    df_fix_log = log_transformation(df_fix)
    print(df_fix.describe())
    print(df_fix_log.describe())

    # normality
    normality_test(df_fix)
    normality_test(df_fix_log)

    # linearity
    pearson = pearson_corr(stdize(df_fix))
    visualize_corr(pearson, df_fix)

    # monotonicity
    spearman = spearman_corr(stdize(df_fix))
    visualize_corr(spearman, df_fix)

    # covariance
    cov = np.cov(stdize(df_fix).T)
    visualize_corr(cov, df_fix)

    # trend-seasonal decomp
    trend, seasonal = decompose(df_fix)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Cooling_Energy"], lw=0.5)
    ax.plot(trend["Cooling_Energy"], lw=2)
    ax.plot(seasonal["Cooling_Energy"], lw=0.5, alpha=0.5)
    ax.grid()

    Ft, Fs = measure_ts_strength(trend, seasonal)
    print(Ft) # strength of trend
    print()
    print(Fs) # strength of seasonality