import os, sys

pardir = os.path.join(f"{os.path.dirname(__file__)}", os.pardir)
sys.path.append(pardir)

import HW1, HW2, HW3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hw1_main():
    wf_final, wp_final, technologies = HW1.helpers.Main()

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"$W_F$")
    for i, kv in enumerate(wf_final.items()):
        k, v = kv
        ax[i].set_title(k)
        ax[i].bar(range(len(v)), v, color='red')
        ax[i].set_xticks(range(len(v)))
        ax[i].set_xticklabels(technologies)
        ax[i].grid()
        ax[i].set_yscale("symlog")
        ax[i].set_ylim(0, 1e8)
    fig.tight_layout()

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"$W_P$")
    for i, kv in enumerate(wp_final.items()):
        k, v = kv
        ax[i].set_title(k)
        ax[i].bar(range(len(v)), v, color='orange')
        ax[i].set_xticks(range(len(v)))
        ax[i].set_xticklabels(technologies)
        ax[i].grid()
        ax[i].set_yscale("symlog")
        ax[i].set_ylim(0, 1e8)
    fig.tight_layout()
    pass


def hw2_a_main():
    series_list, df = HW2.helpers.read_describe()
    df_fix = HW2.helpers.cleansing(df)
    df_fix = HW2.helpers.outliers_quantile(df_fix)
    df_fix_log = HW2.helpers.log_transformation(df_fix)
    # print(df_fix.describe())
    # print(df_fix_log.describe())
    HW2.helpers.normality_test(df_fix)
    HW2.helpers.normality_test(df_fix_log)
    pearson = HW2.helpers.pearson_corr(HW2.helpers.stdize(df_fix))
    HW2.helpers.visualize_corr(pearson, df_fix)

    spearman = HW2.helpers.spearman_corr(HW2.helpers.stdize(df_fix))
    HW2.helpers.visualize_corr(spearman, df_fix)

    cov = np.cov(HW2.helpers.stdize(df_fix).T)
    HW2.helpers.visualize_corr(cov, df_fix)

    trend, seasonal = HW2.helpers.decompose(df_fix)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Cooling_Energy"], lw=0.5)
    ax.plot(trend["Cooling_Energy"], lw=2)
    ax.plot(seasonal["Cooling_Energy"], lw=0.5, alpha=0.5)
    ax.grid()

    Ft, Fs = HW2.helpers.measure_ts_strength(trend, seasonal)
    print(Ft)  # strength of trend
    print()
    print(Fs)  # strength of seasonality


def hw3_a_main():
    series_list, df = HW3.helpers.read_describe_with_timestamp()
    df = df.resample("1H", on="TimeStamp").mean()
    df_fix = HW2.helpers.cleansing(df)
    df_fix = HW2.helpers.outliers_quantile(df_fix)
    df_fix_log = HW2.helpers.log_transformation(df_fix)
    # print(df_fix.describe())
    # print(df_fix_log.describe())
    HW2.helpers.normality_test(df_fix)
    HW2.helpers.normality_test(df_fix_log)
    pearson = HW2.helpers.pearson_corr(HW2.helpers.stdize(df_fix))
    HW2.helpers.visualize_corr(pearson, df_fix)

    spearman = HW2.helpers.spearman_corr(HW2.helpers.stdize(df_fix))
    HW2.helpers.visualize_corr(spearman, df_fix)

    cov = np.cov(HW2.helpers.stdize(df_fix).T)
    HW2.helpers.visualize_corr(cov, df_fix)

    trend, seasonal = HW2.helpers.decompose(df_fix)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Cooling_Energy"], lw=0.5)
    ax.plot(trend["Cooling_Energy"], lw=2)
    ax.plot(seasonal["Cooling_Energy"], lw=0.5, alpha=0.5)
    ax.grid()
    Ft, Fs = HW2.helpers.measure_ts_strength(trend, seasonal)
    print(Ft)  # strength of trend
    print()
    print(Fs)  # strength of seasonality


def hw3_b_main():
    datasets = HW3.helpers_part_b.read_sum_resampled()
    kaggle, ukedc = datasets
    datasets_y = HW3.helpers_part_b.yearly_sum(datasets)
    kaggle_y, ukedc_y = datasets_y
    for i, d in enumerate(kaggle):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(d.index, d["total"], c="red", lw=0.5)
        ax.set_title(
            f"Kaggle-House_{i+1:02d} ({len(d.index):,} samples from {d.shape[1]-2} features)"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Hourly real power consumption (W)")
        ax.set_xlim(d.index[0], d.index[-1])
        ax.set_ylim(0, 5000)
        ax.grid()
        fig.tight_layout()
    for i, d in enumerate(ukedc):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(d.index, d["total"], c="blue", lw=0.5)
        ax.set_title(
            f"UKEDC-House_{i+1:02d} ({len(d.index):,} samples from {d.shape[1]-1} features))"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Hourly real power consumption (W)")
        ax.set_xlim(d.index[0], d.index[-1])
        ax.set_ylim(0, 5000)
        ax.grid()
        fig.tight_layout()
    plt.show()

    years = [2012, 2013, 2014, 2015]
    names = [f"Kaggle-House_{i+1:02d}" for i in range(len(kaggle_y))] + [
        f"UKEDC-House_{i+1:02d}" for i in range(len(ukedc_y))
    ]

    for y in years:
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, d in enumerate(kaggle_y + ukedc_y):
            try:
                ax.bar(i, d[["total"]].loc[f"{y}-12-31"])
            except KeyError:
                continue
        ax.set_xticks(range(len(kaggle_y + ukedc_y)))
        ax.set_xticklabels(names, rotation=-90)
        ax.set_ylabel("Yearly real power consumption (W)")
        ax.set_ylim(1e0, 1e8)
        ax.set_yscale("log")
        ax.set_title(f"Year {y}")
        ax.grid()
        fig.tight_layout()
