import numpy as np
import pandas as pd
import glob, os


def ukedc_to_csv():
    getnum = lambda x: int(x.split("_")[-1].replace(".dat", ""))
    houses = [f"house_{i+1}" for i in range(4)]
    for house in houses:
        # get labels
        with open(f"data_ukedc/{house}/labels.dat") as f:
            names = f.readlines()
            for i, n in enumerate(names):
                names[i] = n.split(" ")[-1].replace("\n", "")
            f.close()
        # to csv with labels and datetime
        os.makedirs(f"data_ukedc/csv/{house}", exist_ok=True)
        for path in sorted(glob.glob(f"data_ukedc/{house}/*.dat")):
            if "labels.dat" not in path:
                d = pd.read_fwf(path, header=None, index_col=False, engine="c")
                datetime = pd.to_datetime(d[0], unit="s")
                d[0] = datetime
                d[1] = (
                    d[1].astype(str).str.replace("\s+", "", regex=True).astype(float)
                )  # remove all fucking whitespaces
                d = d.rename(columns={0: "datetime", 1: names[getnum(path) - 1]})
                d.to_csv(
                    f"data_ukedc/csv/{house}/{getnum(path):02d}-{names[getnum(path) - 1]}.csv",
                    index=False,
                )
                print(
                    f"Saved: data_ukedc/csv/{house}/{getnum(path):02d}-{names[getnum(path) - 1]}.csv"
                )


def ukedc_resample():
    os.makedirs("data_hourly", exist_ok=True)
    houses = [f"house_{i+1}" for i in range(4)]
    for housenum, house in enumerate(houses):
        resampled = []
        for path in sorted(glob.glob(f"data_ukedc/csv/{house}/*.csv")):
            d = pd.read_csv(path)
            _, name = d.columns
            d["datetime"] = pd.to_datetime(d["datetime"])
            d[name] = d[name].astype(float)
            d = d.resample("1H", on="datetime").mean()
            resampled.append(d)
        house_hourly = pd.concat(resampled, axis=1)
        house_hourly.to_csv(f"data_hourly/ukedc_House_{housenum+1:02d}.csv", index=True)
        print(f"Saved: data_hourly/ukedc_House_{housenum+1:02d}.csv")


def kaggle_resample():
    gethousenum = lambda x: int(x.split("/")[-1].replace(".csv", "").split("_")[-1])
    os.makedirs("data_hourly", exist_ok=True)
    for path in sorted(glob.glob(f"data_kaggle/*.csv")):
        d = pd.read_csv(path)
        datetime = pd.to_datetime(d["Time"])
        d.drop(labels="Time", axis=1, inplace=True)
        d["datetime"] = datetime
        d = d.resample("1H", on="datetime").mean()
        d.to_csv(f"data_hourly/kaggle_House_{gethousenum(path):02d}.csv", index=True)
        print(f"Saved: data_hourly/kaggle_House_{gethousenum(path):02d}.csv")


def read_sum_resampled():
    kaggle = []
    ukedc = []
    for p in sorted(glob.glob("data_hourly/*.csv")):
        data = pd.read_csv(p, index_col="datetime")
        data.index = pd.to_datetime(data.index)
        data = data.interpolate(method="nearest")
        data = data.fillna(0)
        assert data.isna().sum().sum() == 0
        if "kaggle" in p:
            data["total"] = data.iloc[:, 2:].sum(axis=1)  # all appliances
            kaggle.append(data)
        elif "ukedc" in p:
            data["total"] = data.iloc[:, 1:].sum(axis=1)
            ukedc.append(data)
    return kaggle, ukedc


def yearly_sum(datasets):
    kaggle, ukedc = datasets
    kaggle_y = []
    ukedc_y = []
    for d in kaggle:
        kaggle_y.append(d[["total"]].resample("1Y").sum())
    for d in ukedc:
        ukedc_y.append(d[["total"]].resample("1Y").sum())
    return kaggle_y, ukedc_y


if __name__ == "__main__":
    # kaggle_resample()
    # ukedc_to_csv()
    # ukedc_resample()
    # datasets = read_sum_resampled()
    # datasets_y = yearly_sum(datasets)
    pass
