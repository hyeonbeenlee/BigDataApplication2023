import numpy as np
import pandas as pd
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from scipy import stats


def _get_filename(string):
    return string.split("/")[-1].replace(".csv", "")


def convert_to_csv():
    """
    WHY xlsx for such a big dataset? HORRIBLY SLOW FORMAT SO GONNA FIX IT
    """
    for idx, path in enumerate(glob.glob("Data/*.xlsx")):
        data = pd.read_excel(path, header=None)
        data.to_csv(path.replace("xlsx", "csv"), index=False)
        print(f"Saved as: {path.replace('xlsx','csv')}")


def read_describe():
    series_list = []
    for idx, path in enumerate(glob.glob("Data/*.csv")):
        data = pd.read_csv(path, names=[_get_filename(path)]).reset_index(
            drop=True
        )  # series
        series_list.append(data)
        print(path)
        print(data.describe())
        print("=" * 100)
    df = pd.concat(series_list, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    return series_list, df


def visualize(df1, df2=None):
    os.makedirs("figures", exist_ok=True)
    for c in df1.columns:
        fig, ax = plt.subplots(2, 1, figsize=(7, 3))
        ax[0].plot(df1[c], lw=0.5, marker="o", mfc="none", ms=2)
        counts, bins = np.histogram(df1[c].to_numpy(), bins=10)
        ax[1].hist(bins[:-1], bins, weights=counts)
        if df2 is not None:
            ax[0].plot(df2[c], lw=0.5, marker="o", mfc="none", ms=2, alpha=0.5)
            counts, bins = np.histogram(df2[c].to_numpy(), bins=10)
            ax[1].hist(bins[:-1], bins, weights=counts, alpha=0.5)
        # ax[1].set_yscale('symlog')
        ax[1].grid()
        fig.suptitle(f"{c}")
        fig.tight_layout()
        fig.savefig(f"figures/{c}.png", dpi=200)
        plt.close(fig)


def cleansing(df):
    df = df.copy()
    print(f"Found {df.isna().sum().sum()} missing values from dataset")
    try:
        assert df.isna().sum().sum() == 0
    except:
        df = df.fillna(0)
    return df


def outliers_quantile(df):
    df = df.copy()
    df_ = df.copy()
    q_low = df.quantile(
        0.001, interpolation="nearest"
    )  # for LogTrans, set higher lower bound
    q_high = df.quantile(0.999, interpolation="nearest")
    for c in df.columns:
        df[c][df[c] <= q_low[c]] = q_low[c]
        df[c][df[c] >= q_high[c]] = q_high[c]
        print(
            f"[{c}] clipping range {df_[c].min():.3f}~{df_[c].max():.3f} to {q_low[c]:.3f}~{q_high[c]:.3f}, processed {df_[c][(df_[c]<=q_low[c]) | (df_[c]>=q_high[c])].shape[0]}/{df.shape[0]} outliers"
        )
    return df


def normality_test(df):
    alpha = 0.05
    print("=" * 100)
    for c in df.columns:
        # stat, p_val = stats.normaltest(df[c].to_numpy().flatten()) # D’Agostino and Pearson’s test
        # stat, p_val = stats.kstest(df[c].to_numpy().flatten(), stats.norm.cdf) # Kolmogorov-Smirnov test
        stat, p_val = stats.shapiro(df[c].to_numpy().flatten())  # Shapiro-Wilk test
        flag = "(Normal)" if p_val >= alpha else "(NOT normal)"
        print(f"{flag}{c}: {p_val:.4e}")
    stat, p_val = stats.shapiro(np.random.randn(df[c].shape[0]))
    flag = "(Normal)" if p_val >= alpha else "(NOT normal)"
    print(f"{flag}Arbitrary np.randn: {p_val:.4e}")
    print("=" * 100)


def log_transformation(df):
    df = df.copy()
    df_log = np.log(df)
    for c in df.columns:
        uniques = np.sort(pd.unique(df_log[c]))
        # For non-logit data
        if len(uniques) != 2:
            df_log[c][df_log[c] == -np.inf] = uniques[1]
        else:
            df_log[c][df_log[c] == -np.inf] = -1
    return df_log


def stdize(df_raw):
    df_raw = df_raw.copy()
    return (df_raw - df_raw.mean()) / df_raw.std()


def istdize(df_converted, df_raw):
    df_converted = df_converted.copy()
    df_raw = df_raw.copy()
    return (df_converted * df_raw.std()) + df_raw.mean()


def pearson_corr(df):
    """
    Measures linearity between variables, [0,1], higher==more correlation
    """
    return np.corrcoef(df.to_numpy().T)


def spearman_corr(df):
    """
    Measures monotonicity between variables, [0,1] higher==more correlation
    """
    return stats.spearmanr(df.to_numpy()).statistic


def visualize_corr(corr, df):
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.matshow(corr)
    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticklabels(df.columns)
    ax.grid()
    fig.tight_layout()
    fig.colorbar(img)
    plt.show()
    print(corr.round(3))
