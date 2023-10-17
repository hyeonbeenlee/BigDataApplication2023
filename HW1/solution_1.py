import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snippets import *


# Eq 1
def wf(ucf, lhv, exf, nex, lt):
    wff = data["feature (ℓ/MWh)"]
    ucf = data["ucf"]
    lhv = data["lhv"]
    exf = data["exf"]
    nex = data["nex"]
    lt = data["lt"]
    return (wff * ucf * lhv) / (exf * nex * lt)


# Eq 2
def wp(wfp, ucf, av, cf, nex, lt):
    wfp = data["feature (ℓ/MWh)"]
    ucf = data["ucf"]
    av = data["av"]
    cf = data["cf"]
    nex = data["nex"]
    lt = data["lt"]
    return (wfp * ucf) / (av * cf * nex * lt)


# Apply STEP 3
def apply_step3(data: pd.DataFrame):
    ucfs = {"Natural gas": 1, "Coal": 1}
    lhvs = {"Natural gas": 52.2, "Coal": 30.2}
    exfs = {"Natural gas": 0.052, "Coal": 0.034}
    lts = {}
    lts.update(zip(["Coal", "Natural gas", "Nuclear", "Geothermal"], [30] * 4))
    lts.update(zip(["PV", "Wind", "CSP"], [15] * 3))
    data["ucf"] = data["type"].map(ucfs)
    data["lhv"] = data["type"].map(lhvs)
    data["exf"] = data["type"].map(exfs)
    data["lt"] = data["type"].map(lts)
    return data


def apply_step4(data: pd.DataFrame):
    np.random.seed(0)
    data[["nex", "av", "cf"]] = np.nan
    data.loc[data["type"] == "Natural gas", "nex"] = np.random.uniform(
        0.218, 0.53, size=data.loc[data["type"] == "Natural gas", "nex"].shape
    )
    return data


# def sample(data:pd.DataFrame, ):

data = pd.read_excel("HW1/data/Power plot data.xlsx").to_numpy()
data_dict = {
    k: v
    for k, v in zip(
        ["category", "type", "cycle", "label4", "label5", "label6", "value"], data.T
    )
}

consumption = data[np.where(data_dict["category"] == "Consumption")]
withdrawal = data[np.where(data_dict["category"] == "Withdrawal")]

# check_unique_values(consumption)
# check_unique_values(withdrawal)
consumption = apply_step3(consumption)
consumption = apply_step4(consumption)
pass
