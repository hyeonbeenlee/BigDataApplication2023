import pandas as pd
import numpy as np
from snippets import *


# Eq 1
def WF(data):
    np.random.seed(0)

    wff = data[:, -1]

    # Assumptions in Step 3
    ucf = np.full_like(data[:, 0], np.nan)
    lhv = np.full_like(data[:, 0], np.nan)
    exf = np.full_like(data[:, 0], np.nan)
    lt = np.full_like(data[:, 0], np.nan)
    nex = np.full_like(data[:, 0], np.nan)

    ucf[np.where(data[:, 1] == "Natural gas")] = 1
    ucf[np.where(data[:, 1] == "Coal")] = 1

    lhv[np.where(data[:, 1] == "Natural gas")] = 52.2
    lhv[np.where(data[:, 1] == "Coal")] = 30.2

    exf[np.where(data[:, 1] == "Natural gas")] = 0.052
    exf[np.where(data[:, 1] == "Coal")] = 0.034

    lt[
        np.where(
            (data[:, 1] == "Coal")
            | (data[:, 1] == "Natural gas")
            | (data[:, 1] == "Nuclear")
            | (data[:, 1] == "Geothermal")
        )
    ] = 30
    lt[
        np.where((data[:, 1] == "PV") | (data[:, 1] == "Wind") | (data[:, 1] == "CSP"))
    ] = 15

    nex[np.where(data[:, 1] == "Coal")] = np.random.uniform(
        0.218, 0.53, size=np.where(data[:, 1] == "Coal")[0].shape
    )
    nex[np.where(data[:, 1] == "Natural gas")] = np.random.uniform(
        0.17, 0.7, size=np.where(data[:, 1] == "Natural gas")[0].shape
    )
    nex[np.where(data[:, 1] == "Nuclear")] = np.random.uniform(
        0.2899, 0.461, size=np.where(data[:, 1] == "Nuclear")[0].shape
    )
    nex[np.where(data[:, 1] == "Geothermal")] = np.random.uniform(
        0.25, 0.83, size=np.where(data[:, 1] == "Geothermal")[0].shape
    )
    nex[np.where(data[:, 1] == "CSP")] = np.random.uniform(
        0.06, 0.597, size=np.where(data[:, 1] == "CSP")[0].shape
    )
    nex[np.where(data[:, 1] == "Wind")] = np.random.uniform(
        0.01, 0.92, size=np.where(data[:, 1] == "Wind")[0].shape
    )
    nex[np.where(data[:, 1] == "PV")] = np.random.uniform(
        0.0251, 0.15, size=np.where(data[:, 1] == "PV")[0].shape
    )

    Wf = (wff * ucf * lhv) / (exf * nex * lt)

    # Assumptions in Step 2
    Wf[np.where((data[:, 2] == "Operating") | (data[:, 2] == "Non-operating"))] = 0
    Wf[np.where(data[:, 1] == "Nuclear")] = 0
    # Wf = Wf[np.where((data[:, 1] == "Natural gas") | (data[:, 1] == "Coal"))]
    # data = data[np.where((data[:, 1] == "Natural gas") | (data[:, 1] == "Coal"))]

    # df['ucf']=ucf
    # df['lhv']=lhv
    # df['exf']=exf
    # df['nex']=nex
    # df['lt']=lt
    return np.nan_to_num(Wf.astype(np.float64))


# Eq 2
def WP(data):
    np.random.seed(0)

    wfp = data[:, -1]

    # Assumptions in Step 3
    ucf = np.full_like(data[:, 0], np.nan)
    av_cf = np.full_like(data[:, 0], np.nan)
    lt = np.full_like(data[:, 0], np.nan)
    nex = np.full_like(data[:, 0], np.nan)

    ucf[np.where(data[:, 1] == "Natural gas")] = 1
    ucf[np.where(data[:, 1] == "Coal")] = 1

    av_cf[np.where(data[:, 1] == "Coal")] = np.random.uniform(
        0.6046, 1, size=np.where(data[:, 1] == "Coal")[0].shape
    )
    av_cf[np.where(data[:, 1] == "Natural gas")] = np.random.uniform(
        0.3476, 1, size=np.where(data[:, 1] == "Natural gas")[0].shape
    )
    av_cf[np.where(data[:, 1] == "Nuclear")] = np.random.uniform(
        0.905, 1, size=np.where(data[:, 1] == "Nuclear")[0].shape
    )
    av_cf[np.where(data[:, 1] == "Geothermal")] = np.random.uniform(
        0.7838, 1, size=np.where(data[:, 1] == "Geothermal")[0].shape
    )
    av_cf[np.where(data[:, 1] == "CSP")] = np.random.uniform(
        0.0448, 1, size=np.where(data[:, 1] == "CSP")[0].shape
    )
    av_cf[np.where(data[:, 1] == "Wind")] = np.random.uniform(
        0.122, 1, size=np.where(data[:, 1] == "Wind")[0].shape
    )
    av_cf[np.where(data[:, 1] == "PV")] = np.random.uniform(
        0.0448, 1, size=np.where(data[:, 1] == "PV")[0].shape
    )

    lt[
        np.where(
            (data[:, 1] == "Coal")
            | (data[:, 1] == "Natural gas")
            | (data[:, 1] == "Nuclear")
            | (data[:, 1] == "Geothermal")
        )
    ] = 30
    lt[
        np.where((data[:, 1] == "PV") | (data[:, 1] == "Wind") | (data[:, 1] == "CSP"))
    ] = 15

    nex[np.where(data[:, 1] == "Coal")] = np.random.uniform(
        0.218, 0.53, size=np.where(data[:, 1] == "Coal")[0].shape
    )
    nex[np.where(data[:, 1] == "Natural gas")] = np.random.uniform(
        0.17, 0.7, size=np.where(data[:, 1] == "Natural gas")[0].shape
    )
    nex[np.where(data[:, 1] == "Nuclear")] = np.random.uniform(
        0.2899, 0.461, size=np.where(data[:, 1] == "Nuclear")[0].shape
    )
    nex[np.where(data[:, 1] == "Geothermal")] = np.random.uniform(
        0.25, 0.83, size=np.where(data[:, 1] == "Geothermal")[0].shape
    )
    nex[np.where(data[:, 1] == "CSP")] = np.random.uniform(
        0.06, 0.597, size=np.where(data[:, 1] == "CSP")[0].shape
    )
    nex[np.where(data[:, 1] == "Wind")] = np.random.uniform(
        0.01, 0.92, size=np.where(data[:, 1] == "Wind")[0].shape
    )
    nex[np.where(data[:, 1] == "PV")] = np.random.uniform(
        0.0251, 0.15, size=np.where(data[:, 1] == "PV")[0].shape
    )

    Wp = (wfp * ucf) / (av_cf * nex * lt)

    # df['ucf']=ucf
    # df['av_cf']=av_cf
    # df['nex']=nex
    # df['lt']=lt
    return np.nan_to_num(Wp.astype(np.float64))


def RandomSum(wf, wp):
    wf = wf.reshape(-1, 1)
    wp = wp.reshape(-1, 1)
    np.random.seed(0)

    # Random values according to Step 4
    rv = np.full((data.shape[0], 10), 0.0)
    fu = np.full((data.shape[0], 10), 0.0)

    rv[np.where(data[:, 1] == "CSP")] = np.random.uniform(
        0.8, 1.2, size=(np.where(data[:, 1] == "CSP")[0].shape[0], 10)
    )
    rv[np.where(data[:, 1] == "Wind")] = np.random.uniform(
        0.8, 1.2, size=(np.where(data[:, 1] == "Wind")[0].shape[0], 10)
    )
    rv[np.where(data[:, 1] == "PV")] = np.random.uniform(
        0.8, 1.2, size=(np.where(data[:, 1] == "PV")[0].shape[0], 10)
    )

    fu[np.where(data[:, 1] == "Coal")] = np.random.uniform(
        1, 1.029, size=(np.where(data[:, 1] == "Coal")[0].shape[0], 10)
    )
    fu[np.where(data[:, 1] == "Natural gas")] = np.random.uniform(
        1, 1.049, size=(np.where(data[:, 1] == "Natural gas")[0].shape[0], 10)
    )
    fu[np.where(data[:, 1] == "Nuclear")] = np.random.uniform(
        1, 1.039, size=(np.where(data[:, 1] == "Nuclear")[0].shape[0], 10)
    )

    Wf = (fu * wf).sum(axis=1)
    Wp = (rv * wp).sum(axis=1)
    return Wf, Wp


# ["category", "type", "cycle", "label4", "label5", "label6", "value"]
data = pd.read_excel("HW1/data/Power plot data.xlsx").to_numpy()
df = pd.read_excel("HW1/data/Power plot data.xlsx")

# consumption = data[np.where(data[:, 0] == "Consumption")]
# withdrawal = data[np.where(data[:, 0] == "Withdrawal")]

wf = WF(data)
wp = WP(data)
wf, wp = RandomSum(wf, wp)

# sum values
consumption = data[np.where(data[:, 0] == "Consumption")]
withdrawal = data[np.where(data[:, 0] == "Withdrawal")]
wf_final = {"consumption": [], "withdrawal": []}
wp_final = {"consumption": [], "withdrawal": []}
for k in wf_final.keys():
    for c in ["Coal", "Natural gas", "Nuclear", "Geothermal", "PV", "Wind", "CSP"]:
        wf_final[k].append(wf[np.where((data[:, 1] == c))].sum())
        wp_final[k].append(wp[np.where((data[:, 1] == c))].sum())
print(wf_final)
print(wp_final)

df["WF"] = wf
df["WP"] = wp
df.to_excel("HW1/result_raw.xlsx", index=False)


# wf_consumption = WF(consumption)
# wf_withdrawal = WF(withdrawal)

# wp_consumption = WP(consumption)
# wp_withdrawal = WP(withdrawal)

# wex_consumption = WEx(wf_consumption, wp_consumption)
# wex_withdrawal = WEx(wf_withdrawal, wp_withdrawal)

pass
