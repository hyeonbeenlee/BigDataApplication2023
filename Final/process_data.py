import os, sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import math
import shutil
import matplotlib.pyplot as plt
import argparse
import glob
from copy import deepcopy
from scipy.interpolate import interp1d
from torch.nn.utils import rnn
from utils.signal import filtbutterworth


sys.path.append(os.getcwd())

from utils.snippets import *
from utils.datareader import readmat
from data.manager import DataManager
from architectures.series import series_decomp


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2eul(R):
    # https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py#L12
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def robot_fkine(jointpos: np.ndarray):
    # body0
    r0 = np.array([0, 0, 0]).reshape(-1, 1)
    A0 = np.eye(3)
    s01_p = np.array([0, 0, 0]).reshape(-1, 1)
    C01 = np.eye(3)

    # body1
    s12_p = np.array([171, 0, 198.5]).reshape(-1, 1)
    C12 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # body2
    s23_p = np.array([921.36, 0, 0]).reshape(-1, 1)
    C23 = np.eye(3)

    # body3
    s34_p = np.array([535.94, 0, 0]).reshape(-1, 1)
    C34 = np.eye(3)

    # body4
    s45_p = np.array([146, 0, 0]).reshape(-1, 1)
    C45 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # body5
    s56_p = np.array([0, 2.38e02, 0]).reshape(-1, 1)
    C56 = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    # body6
    s_t = np.array([0, 0, -262]).reshape(-1, 1)
    s_tc = np.array([0, 0, -107]).reshape(-1, 1)

    # forward kinematics
    time = jointpos[:, 0]
    q = jointpos[:, 1:]
    q2A = lambda q: np.array(
        [[np.cos(q), -np.sin(q), 0], [np.sin(q), np.cos(q), 0], [0, 0, 1]]
    )
    r_tool = np.zeros((q.shape[0], 7))
    for i in range(jointpos.shape[0]):
        # global orientation
        A01_pp = q2A(q[i, 0])
        A12_pp = q2A(q[i, 1])
        A23_pp = q2A(q[i, 2])
        A34_pp = q2A(q[i, 3])
        A45_pp = q2A(q[i, 4])
        A56_pp = q2A(q[i, 5])
        A1 = A0 @ C01 @ A01_pp
        A2 = A1 @ C12 @ A12_pp
        A3 = A2 @ C23 @ A23_pp
        A4 = A3 @ C34 @ A34_pp
        A5 = A4 @ C45 @ A45_pp
        A6 = A5 @ C56 @ A56_pp
        roll, pitch, yaw = rotm2eul(A6)

        # global position
        r1 = r0 + A0 @ s01_p
        r2 = r1 + A1 @ s12_p
        r3 = r2 + A2 @ s23_p
        r4 = r3 + A3 @ s34_p
        r5 = r4 + A4 @ s45_p
        r6 = r5 + A5 @ s56_p
        rtc = r6 + s_tc
        rt = r6 + s_t
        x, y, z = rt.flatten() / 1000
        # rp(i, :) = r6
        r_tool[i, :] = np.array([jointpos[i, 0], x, y, z, roll, pitch, yaw])
        if (i + 1) % 10000 == 0:
            print(f"FK {i+1}/{q.shape[0]}")
    return r_tool


def read(use_filtered: bool = False):
    if not use_filtered:  # raw data
        keyval = {
            "imu": [
                "time",
                "roll",
                "pitch",
                "yaw",
                "orientation_w",
                "roll_angvel",
                "pitch_angvel",
                "yaw_angvel",
                "x_linacc",
                "y_linacc",
                "z_linacc",
            ],
            "force": ["time", "time_acq", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            + [
                f"{F}_calibrated" for F in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            ],  # checked
            "hydraulics_pressure": ["time", "time_acq"]
            + [f"ch{i:02d}" for i in range(1, 33)],
            "diffpressure": ["time"] + [f"diffp_j{i}" for i in range(1, 8)],
            "jointpos": ["time"] + [f"pos_j{i}" for i in range(1, 7)],
            # "depth": ["time", "depth"], # do not use
            "toolpos": ["time", "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"],
        }
    if use_filtered:  # filtered data
        keyval = {
            "imu": [
                "time",
                "roll",
                "pitch",
                "yaw",
                "roll_angvel",
                "pitch_angvel",
                "yaw_angvel",
                "x_linacc",
                "y_linacc",
                "z_linacc",
            ],
            "force": ["time"]
            + [
                f"{F}_calibrated" for F in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            ],  # checked
            "toolpressure": ["time", "hydraulic_pressure"],
            "diffpressure": ["time"] + [f"diffp_j{i}" for i in range(1, 5)],
            "toolpos": ["time", "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"],
            "depth": ["time", "depth"],
        }

    # Read data
    dataset = {}
    for dir, subdirs, files in os.walk("data"):
        # print(dir, subdirs, files)
        if "main_totalSol.m" in files:  # for experiments
            exp_name = dir.split("/")[-1]
            dataset[exp_name] = (
                {} if exp_name not in dataset.keys() else dataset[exp_name]
            )
            for data_name in files:  # for sensor measurements
                for key in keyval.keys():  # check consistency and read
                    if use_filtered:
                        if key in data_name.lower() and "Filt" in data_name:
                            dataset[exp_name][key] = (
                                deepcopy(keyval[key]),
                                list(readmat(f"{dir}/{data_name}").values())[0],
                            )
                    elif not use_filtered:
                        if key in data_name.lower() and "Filt" not in data_name:
                            dataset[exp_name][key] = (
                                deepcopy(keyval[key]),
                                list(readmat(f"{dir}/{data_name}").values())[0],
                            )
                        elif (
                            key == "depth"
                            and key in data_name.lower()
                            and "Filt" in data_name
                        ):
                            dataset[exp_name][key] = (
                                deepcopy(keyval[key]),
                                list(readmat(f"{dir}/{data_name}").values())[0],
                            )
                        elif key == "toolpos" and key not in dataset[exp_name].keys():
                            dataset[exp_name][key] = (
                                deepcopy(keyval[key]),
                                robot_fkine(
                                    list(
                                        readmat(
                                            f"{dir}/knr_uw3_state_jointpos.mat"
                                        ).values()
                                    )[0]
                                ),
                            )
            assert len(dataset[exp_name]) == len(
                keyval.keys()
            ), "Unexpected data content"
    assert len(dataset) == 12, f"Number of experiments is not 12 but {len(dataset)}"

    # validate
    for exp_name in dataset.keys():
        for data_name in keyval.keys():
            assert dataset[exp_name][data_name][-1].shape[1] == len(keyval[data_name])
    sorted_dict = dict(sorted(dataset.items()))
    return sorted_dict


def isDifferent(origin, dataset):
    flags = []
    if type(origin) == dict and type(dataset) == dict:
        for item1, item2 in zip(origin.items(), dataset.items()):
            k1, v1 = item1
            k2, v2 = item2
            flags.append(k1 == k2)
            flags.append(isDifferent(v1, v2))
    elif type(origin) == np.ndarray and type(dataset) == np.ndarray:
        for item1, item2 in zip(origin, dataset):
            flags.append(np.all(item1 == item2))
    else:
        for item1, item2 in zip(origin, dataset):
            flags.append(item1 == item2)
    if np.any([np.any(f) for f in flags]):
        return True  # changed
    else:
        return False  # no changes


def proc_zero_initial_time(dataset: dict):
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]

            # zero initial times
            idx_time = columns.index("time")
            idx_time_acq = columns.index("time_acq") if "time_acq" in columns else None
            data[:, idx_time] -= data[0, idx_time]
            if idx_time_acq is not None:
                data[:, idx_time_acq] -= data[0, idx_time_acq]
    assert isDifferent(origin, dataset)
    return dataset


def proc_uniform_time_steps(dataset: dict):
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]
            idx_time = columns.index("time")
            idx_time_acq = columns.index("time_acq") if "time_acq" in columns else None

            data[:, idx_time] = np.linspace(
                data[:, idx_time][0],
                data[:, idx_time][-1],
                data.shape[0],
                endpoint=True,
            )
            if idx_time_acq is not None:
                data[:, idx_time_acq] = np.linspace(
                    data[:, idx_time_acq][0],
                    data[:, idx_time_acq][-1],
                    data.shape[0],
                    endpoint=True,
                )
    assert isDifferent(origin, dataset)
    return dataset


def proc_correct_offset(dataset: dict):
    origin = deepcopy(dataset)
    force_means = []
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]

            # zero initial times
            idx_time = columns.index("time")
            idx_time_acq = columns.index("time_acq") if "time_acq" in columns else None

            # correct sensor offset except for force
            inits = np.where(data[:, idx_time] <= 5)[0]  # initial 5s
            if not data_name == "force":
                if idx_time_acq is None:
                    data[:, 1:] -= data[inits, 1:].mean(axis=0)
                else:
                    data[:, 2:] -= data[inits, 2:].mean(axis=0)
            elif data_name == "force":
                force_means.append(data[inits, 2:].mean(axis=0))

    # correct FT sensor offset
    force_means = np.stack(force_means, axis=0).mean(axis=0)
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]
            if data_name == "force":
                inits = np.where(data[:, idx_time] <= 5)[0]
                data[:, 2:] -= data[inits, 2:].mean(axis=0)  # remove initial offset
                data[:, 2:] += force_means  # add mean offset
    # visualize force correction
    plot_template(12)
    fig1, ax1 = plt.subplots(3, 2, figsize=(8, 5), sharex=True, sharey=False)
    fig2, ax2 = plt.subplots(3, 2, figsize=(8, 5), sharex=True, sharey=False)
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]
            columns, data_raw = origin[exp_name][data_name]
            if data_name == "force":
                k = 0
                for i in range(3):
                    for j in range(2):
                        ax1[i, j].plot(data[:, k + 8], alpha=0.2)
                        ax2[i, j].plot(data_raw[:, k + 8], alpha=0.2)
                        ax1[i, j].hlines(
                            np.mean(data[:500, k + 8]),
                            0,
                            data.shape[0],
                            color="k",
                            lw=1,
                            zorder=4,
                        )
                        ax2[i, j].hlines(
                            np.mean(data_raw[:500, k + 8]),
                            0,
                            data.shape[0],
                            color="k",
                            lw=1,
                            zorder=4,
                        )
                        # ax1[i, j].set_ylim(-400, -200)
                        # ax2[i, j].set_ylim(-400, -200)
                        ax1[i, j].set_ylabel(columns[k + 8])
                        ax2[i, j].set_ylabel(columns[k + 8])
                        k += 1
    fig1.suptitle("After Bias Correction")
    fig2.suptitle("Raw")
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f"{os.path.dirname(__file__)}/offset_after.png", dpi=300)
    fig2.savefig(f"{os.path.dirname(__file__)}/offset_before.png", dpi=300)
    assert isDifferent(origin, dataset)
    return dataset


def proc_interp_time(dataset: dict, f=20):
    assert f in [20, 100], "Invalid sampling frequency"
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        n_samples = []
        time_lengths = []
        dts = []
        # add interpolators
        for data_name in dataset[exp_name].keys():
            columns, data = dataset[exp_name][data_name]
            # create zeroth polynomial interp
            if "time_acq" in columns:
                interp = interp1d(
                    data[:, 0],
                    data[:, 2:],
                    kind="zero",
                    axis=0,
                    fill_value="extrapolate",
                )
            else:
                interp = interp1d(
                    data[:, 0],
                    data[:, 1:],
                    kind="zero",
                    axis=0,
                    fill_value="extrapolate",
                )
            dataset[exp_name][data_name] = (columns, data, interp)

            n_samples.append(data.shape[0])
            time_lengths.append(np.ptp(data[:, 0]))
            dts.append(np.ptp(data[:, 0]) / data.shape[0])
        # replace to interpolation
        if f == 20:
            time = np.linspace(
                0, time_lengths[np.argmin(n_samples)], np.min(n_samples), endpoint=True
            )
        elif f == 100:
            time = np.linspace(
                0, time_lengths[np.argmin(n_samples)], np.max(n_samples), endpoint=True
            )
        interpolated = [time.reshape(-1, 1)]
        columns_all = ["time"]
        for data_name in dataset[exp_name].keys():
            columns, data, interp = dataset[exp_name][data_name]
            interpolated.append(interp(time))
            if "time" in columns:
                columns.remove("time")
            if "time_acq" in columns:
                columns.remove("time_acq")
            assert interp(time).shape[1] == len(columns)
            columns_all += [data_name + f"_{c}" for c in columns]
        interpolated = np.concatenate(interpolated, axis=1)

        # save
        dataset[exp_name] = pd.DataFrame(interpolated, columns=columns_all)
    assert isDifferent(origin, dataset)
    return dataset


def proc_match_coords(dataset: dict):
    # Match all coordinates to global origin coordiate
    # IMU Sensors == Global coordinate
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in origin[exp_name].keys():
            columns, data = origin[exp_name][data_name]

            if data_name == "force":
                dataset[exp_name][data_name][1][:, columns.index("Fx")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Fy")]
                dataset[exp_name][data_name][1][:, columns.index("Fy")] = -origin[
                    exp_name
                ][data_name][1][:, columns.index("Fx")]
                dataset[exp_name][data_name][1][:, columns.index("Fz")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Fz")]
                dataset[exp_name][data_name][1][:, columns.index("Mx")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("My")]
                dataset[exp_name][data_name][1][:, columns.index("My")] = -origin[
                    exp_name
                ][data_name][1][:, columns.index("Mx")]
                dataset[exp_name][data_name][1][:, columns.index("Mz")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Mz")]
                dataset[exp_name][data_name][1][
                    :, columns.index("Fx_calibrated")
                ] = origin[exp_name][data_name][1][:, columns.index("Fy_calibrated")]
                dataset[exp_name][data_name][1][
                    :, columns.index("Fy_calibrated")
                ] = -origin[exp_name][data_name][1][:, columns.index("Fx_calibrated")]
                dataset[exp_name][data_name][1][
                    :, columns.index("Fz_calibrated")
                ] = origin[exp_name][data_name][1][:, columns.index("Fz_calibrated")]
                dataset[exp_name][data_name][1][
                    :, columns.index("Mx_calibrated")
                ] = origin[exp_name][data_name][1][:, columns.index("My_calibrated")]
                dataset[exp_name][data_name][1][
                    :, columns.index("My_calibrated")
                ] = -origin[exp_name][data_name][1][:, columns.index("Mx_calibrated")]
                dataset[exp_name][data_name][1][
                    :, columns.index("Mz_calibrated")
                ] = origin[exp_name][data_name][1][:, columns.index("Mz_calibrated")]

            # elif data_name == "toolpos":
            #     dataset[exp_name][data_name][1][:, columns.index("pos_x")] = origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("pos_x")]
            #     dataset[exp_name][data_name][1][:, columns.index("pos_y")] = -origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("pos_y")]
            #     dataset[exp_name][data_name][1][:, columns.index("pos_z")] = -origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("pos_z")]
            #     dataset[exp_name][data_name][1][:, columns.index("roll")] = origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("roll")]
            #     dataset[exp_name][data_name][1][:, columns.index("pitch")] = -origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("pitch")]
            #     dataset[exp_name][data_name][1][:, columns.index("yaw")] = -origin[
            #         exp_name
            #     ][data_name][1][:, columns.index("yaw")]
    assert isDifferent(origin, dataset)
    return dataset


def proc_filt(dataset: dict, cutoff=13.5):
    # Match all coordinates to global origin coordiate
    # IMU Sensors == Global coordinate
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in origin[exp_name].keys():
            columns, data = origin[exp_name][data_name]
            if not data_name == "hydraulics_pressure":
                # if data_name == "force":
                data[:, 2:] = filtbutter(
                    data[:, 2:],
                    cutoff=cutoff,
                    timestep=data[1, 0] - data[0, 0],
                    order=20,
                    mode="low",
                )
    assert isDifferent(origin, dataset)
    return dataset


def proc_outliers(dataset: dict):
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in origin[exp_name].keys():
            columns, data = origin[exp_name][data_name]
            if data_name == "hydraulics_pressure":
                data[:, 2] = np.where(data[:, 2] > 20, 0, data[:, 2])
                data[:, 2] = np.where(data[:, 2] < 0, 0, data[:, 2])
    assert isDifferent(origin, dataset)
    return dataset


def postprocess(f: int = 20, use_filtered: bool = False):
    dataset = read(use_filtered=use_filtered)
    dataset = proc_zero_initial_time(dataset)
    dataset = proc_uniform_time_steps(dataset)
    dataset = proc_correct_offset(dataset)
    dataset = proc_filt(dataset, cutoff=10)
    dataset = proc_outliers(dataset)
    dataset = proc_match_coords(dataset)
    dataset = proc_interp_time(dataset, f=f)

    return dataset


def to_csv(dataset: dict, tag: str = ""):
    path = "data/Data_Feb2023/processed_Jul2023"
    os.makedirs(path, exist_ok=True)
    for key in dataset.keys():
        Path = f"{path}/{key}_{tag}" if tag != "" else f"{path}/{key}"
        dataset[key].to_csv(f"{Path}.csv", index=False)
        print(f"{Path}.csv")


def to_npz(seq_len_i: int = 20, seq_len_o: int = 20, forecast=True):
    # for d in os.scandir("data/Data_Feb2023/processed_Jul2023"):
    #     if d.is_dir():
    #         shutil.rmtree(d.path)

    print(f"{len(DataManager.inputs)} inputs, {len(DataManager.outputs)} outputs")

    for k, v in DataManager.files.items():
        for datapath in v:
            data = pd.read_csv(datapath)
            input_data = data[DataManager.inputs].to_numpy()
            output_data = data[DataManager.outputs].to_numpy()
            count = 1

            if forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023_forecast/{k}_{os.path.basename(datapath).replace('.csv','')}"
            elif not forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023/{k}_{os.path.basename(datapath).replace('.csv','')}"
            os.makedirs(subdir, exist_ok=True)
            # initial sequences (incomplete seqs+complete initial seq)
            for t in range(seq_len_i):
                input_zeropad = np.zeros((seq_len_i - (t + 1), input_data.shape[1]))
                if forecast:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = output_data[t + 1 : t + 1 + seq_len_o, :]
                else:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = output_data[: t + 1, :]
                input_sequence = np.concatenate([input_zeropad, input_sequence], axis=0)
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,  # LC
                )
                count += 1
                print(f"{subdir}/ioseq_{count:05d}.npz")

            # running time sequences
            for t in range(input_data.shape[0] - seq_len_i - seq_len_o):
                if forecast:
                    input_sequence = input_data[t + seq_len_i : t + 2 * seq_len_i, :]
                    output_sequence = output_data[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                else:
                    input_sequence = input_data[t + 1 : t + 1 + seq_len_i, :]
                    output_sequence = output_data[t + 1 : t + 1 + seq_len_i, :]
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,  # LC
                )
                count += 1
                print(f"{subdir}/ioseq_{count:05d}.npz")


def to_npz_decomp(
    seq_len_i: int = 20,
    seq_len_o: int = 20,
    forecast=True,
    cutoff=1,
):
    # for d in os.scandir("data/Data_Feb2023/processed_Jul2023"):
    #     if d.is_dir():
    #         shutil.rmtree(d.path)

    print(f"{len(DataManager.inputs)} inputs, {len(DataManager.outputs)} outputs")
    total_time = 0
    total_length = 0
    for k, v in DataManager.files.items():
        for datapath in v:
            data = pd.read_csv(datapath)
            total_time += data["time"].iloc[-1] - data["time"].iloc[0]
            total_length += data.shape[0]
            dt = total_time / total_length

            input_data = data[DataManager.inputs].to_numpy()
            output_data = data[DataManager.outputs].to_numpy()

            # FFT Filtering
            output_fft = np.fft.rfft(output_data, axis=0)
            output_freq = np.fft.rfftfreq(n=output_data.shape[0], d=1 / 20).reshape(
                -1, 1
            )
            kernel = output_freq <= 1
            output_trend = np.fft.irfft(
                kernel * output_fft, n=output_data.shape[0], axis=0
            )

            # output_trend = filtbutterworth(
            #     output_data, cutoff=cutoff, timestep=dt, order=20, mode="low"
            # )
            output_res = output_data - output_trend

            plot_template(13)
            fig, ax = plt.subplots(6, 1, figsize=(9, 8), sharex=True)
            ylabels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i in range(6):
                ax[i].plot(output_data[:, i], c="black", lw=1)
                ax[i].plot(output_res[:, i], c="green", alpha=0.6, lw=0.5)
                ax[i].plot(output_trend[:, i], c="gold", alpha=0.8, lw=1)
                ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel("Time steps")
            fig.tight_layout()
            fig.savefig(
                f"data/Data_Feb2023/processed_Jul2023_forecast_decomp/decomp_{os.path.basename(datapath).replace('.csv','.png')}",
                dpi=200,
            )

            count = 1

            if forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023_forecast_decomp/{k}_{os.path.basename(datapath).replace('.csv','')}"
            elif not forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023_decomp/{k}_{os.path.basename(datapath).replace('.csv','')}"
            os.makedirs(subdir, exist_ok=True)
            # initial sequences (incomplete seqs+complete initial seq)
            for t in range(seq_len_i):
                if forecast:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = output_data[t + 1 : t + 1 + seq_len_o, :]
                    output_res_sequence = output_res[t + 1 : t + 1 + seq_len_o, :]
                    output_trend_sequence = output_trend[t + 1 : t + 1 + seq_len_o, :]
                else:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = output_data[: t + 1, :]
                    output_res_sequence = output_res[: t + 1, :]
                    output_trend_sequence = output_trend[: t + 1, :]
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,
                    output_res_sequence,
                    output_trend_sequence,  # LC
                )
                count += 1
                print(f"{subdir}/ioseq_{count:05d}.npz")

            # running time sequences
            for t in range(input_data.shape[0] - seq_len_i - seq_len_o):
                if forecast:
                    input_sequence = input_data[t + seq_len_i : t + 2 * seq_len_i, :]
                    output_sequence = output_data[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                    output_res_sequence = output_res[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                    output_trend_sequence = output_trend[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                else:
                    input_sequence = input_data[t + 1 : t + 1 + seq_len_i, :]
                    output_sequence = output_data[t + 1 : t + 1 + seq_len_i, :]
                    output_res_sequence = output_res[t + 1 : t + 1 + seq_len_i, :]
                    output_trend_sequence = output_trend[t + 1 : t + 1 + seq_len_i, :]
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,
                    output_res_sequence,
                    output_trend_sequence,  # LC
                )
                count += 1
                print(f"{subdir}/ioseq_{count:05d}.npz")


def to_npz_decomp_fixres(
    seq_len_i: int = 20,
    seq_len_o: int = 20,
    forecast=True,
    cutoff=1,
):
    # for d in os.scandir("data/Data_Feb2023/processed_Jul2023"):
    #     if d.is_dir():
    #         shutil.rmtree(d.path)

    print(f"{len(DataManager.inputs)} inputs, {len(DataManager.outputs)} outputs")

    os.makedirs(
        "data/Data_Feb2023/processed_Jul2023_forecast_decomp_fixres", exist_ok=True
    )

    # Compute RMS
    names = []
    output_res_rms_values = []
    for k, v in DataManager.files.items():
        for datapath in v:
            # k = "temp"
            # for datapath in sorted(
            #     DataManager.files["train"]
            #     + DataManager.files["valid"]
            #     + DataManager.files["test"]
            # ):
            data = pd.read_csv(datapath)
            input_data = data[DataManager.inputs].to_numpy()
            output_data = data[DataManager.outputs].to_numpy()

            # FFT Filtering
            output_fft = np.fft.rfft(output_data, axis=0)
            output_freq = np.fft.rfftfreq(n=output_data.shape[0], d=1 / 20).reshape(
                -1, 1
            )
            kernel = output_freq <= cutoff
            output_trend = np.fft.irfft(
                kernel * output_fft, n=output_data.shape[0], axis=0
            )

            # output_trend = filtbutterworth(
            #     output_data, cutoff=cutoff, timestep=dt, order=20, mode="low"
            # )

            # Moving average filtering
            # kernel_size = 20
            # ret = np.cumsum(output_data, axis=0)
            # ret[kernel_size:] = ret[kernel_size:] - ret[:-kernel_size]
            # output_trend = ret[kernel_size - 1 :] / kernel_size
            # pad1 = np.full((kernel_size // 2, output_data.shape[1]), output_trend[0])
            # pad2 = np.full(
            #     (kernel_size // 2 - 1, output_data.shape[1]), output_trend[-1]
            # )
            # output_trend = np.concatenate([pad1, output_trend, pad2], axis=0)

            output_res = output_data - output_trend

            init_idx = (data["time"] <= 5).to_numpy()
            names.append(datapath)
            # output_res_rms_values.append(
            #     np.sqrt(np.mean(np.square(output_res[init_idx]), axis=0))
            # ) # apply for initial time only
            output_res_rms_values.append(
                np.sqrt(np.mean(np.square(output_res), axis=0))
            )  # apply for total time
            pass
    output_res_rms_values = np.stack(output_res_rms_values, axis=0)
    output_res_rms_mean = np.mean(output_res_rms_values, axis=0)
    pass

    total_time = 0
    total_length = 0
    output_res_rms_values_fixed = []
    for k, v in DataManager.files.items():
        for datapath in sorted(v):
            # k = "temp"
            # for datapath in sorted(
            #     DataManager.files["train"]
            #     + DataManager.files["valid"]
            #     + DataManager.files["test"]
            # ):
            data = pd.read_csv(datapath)
            total_time += data["time"].iloc[-1] - data["time"].iloc[0]
            total_length += data.shape[0]
            dt = total_time / total_length

            input_data = data[DataManager.inputs].to_numpy()
            output_data = data[DataManager.outputs].to_numpy()

            # FFT Filtering
            output_fft = np.fft.rfft(output_data, axis=0)
            output_freq = np.fft.rfftfreq(n=output_data.shape[0], d=1 / 20).reshape(
                -1, 1
            )
            kernel = output_freq <= cutoff
            output_trend = np.fft.irfft(
                kernel * output_fft, n=output_data.shape[0], axis=0
            )

            # output_trend = filtbutterworth(
            #     output_data, cutoff=cutoff, timestep=dt, order=20, mode="low"
            # )

            # Moving average filtering
            # kernel_size = 20
            # ret = np.cumsum(output_data, axis=0)
            # ret[kernel_size:] = ret[kernel_size:] - ret[:-kernel_size]
            # output_trend = ret[kernel_size - 1 :] / kernel_size
            # pad1 = np.full((kernel_size // 2, output_data.shape[1]), output_trend[0])
            # pad2 = np.full(
            #     (kernel_size // 2 - 1, output_data.shape[1]), output_trend[-1]
            # )
            # output_trend = np.concatenate([pad1, output_trend, pad2], axis=0)

            output_res = output_data - output_trend

            # Scaling residuals to measured avaerage
            init_idx = (data["time"] <= 5).to_numpy()
            # res_rms = np.sqrt(np.mean(np.square(output_res[init_idx]), axis=0)) # initial time rms
            res_rms = np.sqrt(np.mean(np.square(output_res), axis=0))  # total time rms
            res_scale = output_res_rms_mean / res_rms
            output_res = output_res * res_scale
            output_res_rms_values_fixed.append(
                np.sqrt(np.mean(np.square(output_res), axis=0))
            )

            # Concat and save
            concatenated = pd.DataFrame(
                np.concatenate(
                    [input_data, output_trend + output_res, output_trend, output_res],
                    axis=1,
                ),
                columns=DataManager.inputs
                + [f"{o}" for o in DataManager.outputs]
                + [f"{o}_trend" for o in DataManager.outputs]
                + [f"{o}_res" for o in DataManager.outputs],
            )
            concatenated.to_csv(
                datapath.replace("Jul2023", "Jul2023_forecast_decomp_fixres"),
                index=False,
            )

            pass
            plot_template(13)
            fig, ax = plt.subplots(6, 1, figsize=(10, 10))
            for i in range(6):
                ax[i].plot(output_data[:, i], c="black", lw=1)
                ax[i].plot(output_res[:, i], c="green", alpha=0.6, lw=0.5)
                ax[i].plot(output_trend[:, i], c="gold", alpha=0.8, lw=1)
            fig.tight_layout()
            fig.savefig(
                f"data/Data_Feb2023/processed_Jul2023_forecast_decomp_fixres/{os.path.basename(datapath).replace('.csv','.png')}",
                dpi=200,
            )

            count = 1

            if forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023_forecast_decomp_fixres/{k}_{os.path.basename(datapath).replace('.csv','')}"
            elif not forecast:
                subdir = f"data/Data_Feb2023/processed_Jul2023_decomp_fixres/{k}_{os.path.basename(datapath).replace('.csv','')}"
            os.makedirs(subdir, exist_ok=True)
            # initial sequences (incomplete seqs+complete initial seq)
            for t in range(seq_len_i):
                if forecast:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = (output_trend + output_res)[
                        t + 1 : t + 1 + seq_len_o, :
                    ]
                    output_res_sequence = output_res[t + 1 : t + 1 + seq_len_o, :]
                    output_trend_sequence = output_trend[t + 1 : t + 1 + seq_len_o, :]
                else:
                    input_sequence = input_data[: t + 1, :]
                    output_sequence = (output_trend + output_res)[: t + 1, :]
                    output_res_sequence = output_res[: t + 1, :]
                    output_trend_sequence = output_trend[: t + 1, :]
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,
                    output_res_sequence,
                    output_trend_sequence,  # LC
                )
                count += 1
                if count % 10000 == 0:
                    print(f"{subdir}/ioseq_{count:05d}.npz")

            # running time sequences
            for t in range(input_data.shape[0] - seq_len_i - seq_len_o):
                if forecast:
                    input_sequence = input_data[t + seq_len_i : t + 2 * seq_len_i, :]
                    output_sequence = (output_trend + output_res)[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                    output_res_sequence = output_res[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                    output_trend_sequence = output_trend[
                        t + 2 * seq_len_i : t + 2 * seq_len_i + seq_len_o, :
                    ]
                else:
                    input_sequence = input_data[t + 1 : t + 1 + seq_len_i, :]
                    output_sequence = (output_trend + output_res)[
                        t + 1 : t + 1 + seq_len_i, :
                    ]
                    output_res_sequence = output_res[t + 1 : t + 1 + seq_len_i, :]
                    output_trend_sequence = output_trend[t + 1 : t + 1 + seq_len_i, :]
                np.savez(
                    f"{subdir}/ioseq_{count:05d}.npz",
                    input_sequence,
                    output_sequence,
                    output_res_sequence,
                    output_trend_sequence,  # LC
                )
                count += 1
                if count % 10000 == 0:
                    print(f"{subdir}/ioseq_{count:05d}.npz")
    output_res_rms_values_fixed = np.stack(output_res_rms_values_fixed, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(9, 5))
    im = ax[0].matshow(
        output_res_rms_values,
        cmap="rainbow",
        vmax=np.max(output_res_rms_values),
        vmin=np.min(output_res_rms_values),
    )
    im2 = ax[1].matshow(
        output_res_rms_values_fixed,
        cmap="rainbow",
        vmax=np.max(output_res_rms_values),
        vmin=np.min(output_res_rms_values),
    )
    ax[0].set_xticks(np.arange(6))
    ax[0].set_xticklabels(DataManager.outputs, rotation=90)
    ax[1].set_xticks(np.arange(6))
    ax[1].set_xticklabels(DataManager.outputs, rotation=90)
    ax[0].set_yticks(np.arange(len(names)))
    ax[0].set_yticklabels([n.split("/")[-1].replace("csv", "") for n in names])
    ax[1].set_yticks(np.arange(len(names)))
    ax[1].set_yticklabels([])
    cbar1 = fig.colorbar(im, ax=ax[0], label="RMS")
    cbar2 = fig.colorbar(im2, ax=ax[1], label="RMS")
    cbar1.set_label("RMS", rotation=270, labelpad=20)
    cbar2.set_label("RMS", rotation=270, labelpad=20)
    fig.tight_layout()
    fig.savefig(
        f"data/Data_Feb2023/processed_Jul2023_forecast_decomp_fixres/rms_diff.png",
        dpi=200,
    )

    pass


def garbage():
    datalist = glob.glob(
        f"data/Data_Feb2023/processed_Jul2023_forecast_decomp_fixres/*.csv"
    )
    data = pd.read_csv(datalist[0])
    data_garbage = data.copy()
    data_garbage = data_garbage.to_numpy().flatten()
    random_idx = np.random.randint(
        0, data_garbage.shape[0], size=data_garbage.shape[0] // 10
    )
    data_garbage[random_idx] = np.nan
    data_garbage = pd.DataFrame(
        data_garbage.reshape(-1, data.shape[1]), columns=data.columns
    )
    time = np.arange(0, 1 / 20 * data.shape[0], 1 / 20)
    freq = np.fft.rfftfreq(n=time.shape[0], d=1 / 20)
    data_garbage = data_garbage.interpolate(method="linear", limit_direction="both")
    trend_src = data[DataManager.outputs].to_numpy()
    trend_garbage = data_garbage[DataManager.outputs].to_numpy()
    fft_src = np.fft.rfft(trend_src, axis=0)
    fft_garbage = np.fft.rfft(trend_garbage, axis=0)
    kernel = (freq <= 1).reshape(-1, 1)
    trend_src = np.fft.irfft(kernel * fft_src, n=trend_src.shape[0], axis=0)
    trend_garbage = np.fft.irfft(kernel * fft_garbage, n=trend_garbage.shape[0], axis=0)

    # visualize
    plot_template(13)
    fig, ax = plt.subplots(3, 2, figsize=(7, 7), sharex=True)
    ax = ax.flatten()
    for i, f in enumerate(DataManager.outputs):
        ax[i].plot(time, trend_src[:, i], c="black", lw=1.2, label="Clean")
        ax[i].plot(
            time,
            trend_garbage[:, i],
            c="green",
            lw=0.8,
            ls="dashed",
            label="Contaminated",
        )
        ax[i].set_ylabel(DataManager.outputs[i])
        ax[i].set_xlabel("Time [sec]")
    increase_leglw(ax[0].legend(loc=2, fontsize=10))
    fig.tight_layout()
    fig.savefig("figures/garbage.png", dpi=200)
    pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="test")
    # parser.add_argument(
    #     "--f",
    #     type=int,
    #     default=20,
    #     help="Sampling frequency to use. Supports [20, 100]. Defaults to 20.",
    # )
    # parser.add_argument(
    #     "--seqlen_i",
    #     type=int,
    #     default=20,
    #     help="Input sequence window size. Defaults to 20.",
    # )
    # parser.add_argument(
    #     "--seqlen_o",
    #     type=int,
    #     default=20,
    #     help="Output sequence window size. Defaults to 20.",
    # )
    # parser.add_argument(
    #     "--forecast",
    #     type=bool,
    #     default=True,
    #     help="Whether to build forecasting dataset. If true, input[:t],output[t:2*t]. Else, input[:t],output[:,t]. Defaults to True.",
    # )
    # args = parser.parse_args()

    # dataset = postprocess(args.f)
    # to_csv(dataset)
    # to_npz(
    #     seq_len_i=args.seqlen_i,
    #     seq_len_o=args.seqlen_o,
    #     forecast=args.forecast,
    # )
    # todo: test sequence windowing codes

    # to_npz_decomp(
    #     seq_len_i=args.seqlen_i,
    #     seq_len_o=args.seqlen_o,
    #     forecast=args.forecast,
    #     cutoff=1,
    # )

    # dataset_raw = read()
    # dataset_raw = proc_zero_initial_time(dataset_raw)
    # dataset_raw = proc_uniform_time_steps(dataset_raw)
    # dataset_raw = proc_correct_offset(dataset_raw)
    # dataset_raw = proc_filt(dataset_raw, cutoff=10)
    # dataset_raw = proc_outliers(dataset_raw)
    # dataset_raw = proc_match_coords(dataset_raw)
    # dataset_raw = proc_interp_time(dataset_raw, f=20)
    # to_csv(dataset_raw, tag='raw')
    to_npz_decomp_fixres(seq_len_i=20, seq_len_o=20, forecast=True, cutoff=1)

    # to_npz(seq_len_i=20, seq_len_o=20, forecast=False)
    # to_npz(seq_len_i=20, seq_len_o=20, forecast=True)
    # to_npz_decomp(seq_len_i=20, seq_len_o=20, forecast=True, cutoff=1)
    # dataset = postprocess(f=20)
    # to_csv(dataset)
    pass
