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


# Eq 1
def WF(data):
    np.random.seed(0)

    wff = data[:, -1]

    # Assumptions in Step 3
    ucf = np.full_like(data[:, 0], 1.0)  # Unit conversion factors to 1
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

    # Sampled from uniform dist. with range given in Table 3
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

    """
    wff: given in excel
    ucf, lhv, exf, lt: given in Step 2
    nex: sampled from Table 3
    """
    Wf = (wff * ucf * lhv) / (exf * nex * lt)

    # Assumptions in Step 2
    Wf[np.where((data[:, 2] == "Operating") | (data[:, 2] == "Non-operating"))] = 0
    Wf[np.where(data[:, 1] == "Nuclear")] = 0
    # Wf = Wf[np.where((data[:, 1] == "Natural gas") | (data[:, 1] == "Coal"))]
    # data = data[np.where((data[:, 1] == "Natural gas") | (data[:, 1] == "Coal"))]

    return np.nan_to_num(Wf.astype(np.float64))

# Eq 2
def WP(data):
    np.random.seed(0)

    wfp = data[:, -1]

    # Assumptions in Step 3
    ucf = np.full_like(data[:, 0], 1.0)  # Unit conversion factors to 1
    av_cf = np.full_like(data[:, 0], 0.0)
    lt = np.full_like(data[:, 0], 0.0)
    nex = np.full_like(data[:, 0], 0.0)

    ucf[np.where(data[:, 1] == "Natural gas")] = 1
    ucf[np.where(data[:, 1] == "Coal")] = 1

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

    # Sampled from uniform distribution in range given in Table 3
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

    return np.nan_to_num(Wp.astype(np.float64))

def RandomWeightedSum(wf, wp, data):
    wf = wf.reshape(-1, 1)
    wp = wp.reshape(-1, 1)
    # np.random.seed(0)

    # Random values according to Step 4
    rv = np.ones((data.shape[0], 10)).astype(np.float64)
    fu = np.ones((data.shape[0], 10)).astype(np.float64)

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

def Main(verbose: bool = True):
    # Used pandas only for reading xlsx file
    data = pd.read_excel("HW1/data/Power plot data.xlsx").to_numpy()
    
    technologies = ["Coal", "Natural gas", "Nuclear", "Geothermal", "PV", "Wind", "CSP"]

    wf = WF(data)
    wp = WP(data)
    wf, wp = RandomWeightedSum(wf, wp, data)

    # sum values
    wf_final = {"Consumption": [], "Withdrawal": []}
    wp_final = {"Consumption": [], "Withdrawal": []}
    for k in wf_final.keys():
        for c in technologies:
            wf_final[k].append(
                wf[np.where((data[:, 1] == c) & (data[:, 0] == k))].sum().round(2)
            )
            wp_final[k].append(
                wp[np.where((data[:, 1] == c) & (data[:, 0] == k))].sum().round(2)
            )

    # Print
    if verbose:
        index = ["WP", "WF"]
        for idx_v, value in enumerate([wp_final.items(), wf_final.items()]):
            for k, v in value:
                for idx_t, t in enumerate(technologies):
                    print(f"{index[idx_v]}_{k}_{t}: {v[idx_t]}")
                print('='*100)
    return wf_final, wp_final, technologies