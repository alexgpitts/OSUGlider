# import Data from CDIP
import wave
from helper_functions import (Data, Rolling_mean, calcPSD, wcalcPSD, wfft, Plotter)
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def process(fn: str, args: ArgumentParser) -> None:

    # fft perameters
    window_type = "hann"

    # call master data function to extract all our data from the .nc file
    data = Data(args.nc[0])
    outputs = []

    # time bounds holds an array starting and ending times for each analisis block
    time_bounds = data["time-bounds"]
    time = data["time"]

    # lists of upper, lower, and midpoint frequencies for banding
    freq_bounds = data["freq"]["bounds"]
    freq_lower = freq_bounds["lower"]
    freq_upper = freq_bounds["upper"]
    freq_midpoints = freq_bounds["joint"].mean(axis=1)

    # print(1/freq_midpoints)
    # print(data["freq"]["bandwidth"] )
    # exit(0)

    # loop runs through every analisis block, displaying output calculations
    # and graphs at the end of each loop run
    for i in range(len(time_bounds["lower"])):

        time_lower = time_bounds["lower"][i]
        time_upper = time_bounds["upper"][i]

        # bit mask so as to select only within the bounds of one lower:upper range pair
        select = np.logical_and(
            time >= time_lower,
            time <= time_upper
        )
        # print(data["time"][select])
        # exit(0)

        # use select to filter to select the acc data corresponding to the current block
        # time = data["time"][select]
        W = 2
        acc = {
            "x": Rolling_mean(data["acc"]["x"][select], W),      # x is northwards
            "y": Rolling_mean(data["acc"]["y"][select], W),      # y is eastwards
            "z": Rolling_mean(data["acc"]["z"][select], W)       # z is upwards
            # "x": data["acc"]["x"][select],      # x is northwards
            # "y": data["acc"]["y"][select],      # y is eastwards
            # "z": data["acc"]["z"][select],       # z is upwards
        }
        # for i in acc["x"]:
        #     print(i)
        # input("Press enter")
        # continue
        

        # preform FFT on block
        FFT = {
            "x": np.fft.rfft(acc["x"], n=acc["z"].size),  # northwards
            "y": np.fft.rfft(acc["y"], n=acc["z"].size),  # eastwards
            "z": np.fft.rfft(acc["z"], n=acc["z"].size),  # upwards
        }

        # preform FFT on block using welch mothod
        wFFT = {
            "x": wfft(acc["x"], 2**8, window_type),
            "y": wfft(acc["y"], 2**8, window_type),
            "z": wfft(acc["z"], 2**8, window_type),
        }

        # Calculate PSD of data from normal FFT
        PSD = {
            # imaginary part is zero
            "xx": calcPSD(FFT["x"], FFT["x"], data["frequency"], "boxcar").real,
            "yy": calcPSD(FFT["y"], FFT["y"], data["frequency"], "boxcar").real,
            "zz": calcPSD(FFT["z"], FFT["z"], data["frequency"], "boxcar").real,

            "xy": calcPSD(FFT["x"], FFT["y"], data["frequency"], "boxcar"),
            "zx": calcPSD(FFT["z"], FFT["x"], data["frequency"], "boxcar"),
            "zy": calcPSD(FFT["z"], FFT["y"], data["frequency"], "boxcar"),
        }

        # calculate PSD on output from welch method FFT
        wPSD = {
            "xx": wcalcPSD(wFFT["x"], wFFT["x"], data["frequency"], window_type).real,
            "yy": wcalcPSD(wFFT["y"], wFFT["y"], data["frequency"], window_type).real,
            "zz": wcalcPSD(wFFT["z"], wFFT["z"], data["frequency"], window_type).real,

            "freq_space": np.fft.rfftfreq(wFFT["z"][0].size*2-1, 1/data["frequency"])
        }

        # frequency space for plotting FFT
        freq_space = np.fft.rfftfreq(acc["z"].size, 1/data["frequency"])

        # bit mask so as to select only within the bounds of one lower:upper range pair
        freq_select = np.logical_and(
            np.less_equal.outer(freq_lower, freq_space),
            np.greater_equal.outer(freq_upper, freq_space)
        )

        count = freq_select.sum(axis=1)

        # Preform Baniding on the PSD. Averages the data withen each bin.
        Band = {
            "xx": (freq_select * PSD["xx"]).sum(axis=1) / count,
            "yy": (freq_select * PSD["yy"]).sum(axis=1) / count,
            "zz": (freq_select * PSD["zz"]).sum(axis=1) / count,

            "xy": (freq_select * PSD["xy"]).sum(axis=1) / count,
            "zx": (freq_select * PSD["zx"]).sum(axis=1) / count,

            "zy": (freq_select * PSD["zy"]).sum(axis=1) / count,
        }

        print("Processing Block {0}".format(i))

        ##########################################
        # Calculations
        ####################### ##################

        Output = {}
        outputs.append(Output)

        ##########################################
        # Calculations using the banded method
        ##########################################

        def banded(run: bool):
            if run == False:
                return
            a0 = Band["zz"] / \
                np.square(np.square(2 * np.pi * freq_midpoints))

            m0 = (a0 * data["freq"]["bandwidth"]).sum()
            m1 = (a0*freq_midpoints*data["freq"]["bandwidth"]).sum()
            mm1 = (a0/freq_midpoints*data["freq"]["bandwidth"]).sum()
            te = mm1/m0  # mean energy period

            m2 = (a0*np.square(freq_midpoints)
                  * data["freq"]["bandwidth"]).sum()

            tp = 1/freq_midpoints[a0.argmax()]

            denom = np.sqrt(Band["zz"] * (Band["xx"] + Band["yy"]))
            a1 = Band["zx"].imag / denom
            b1 = -Band["zy"].imag / denom
            denom = Band["xx"] + Band["yy"]

            dp = np.arctan2(
                b1[a0.argmax()], a1[a0.argmax()])  # radians

            Output["banded"] = {
                "Hs": 4 * np.sqrt(m0),
                "Ta": m0/m1,
                "Tp": tp,  # peak wave period
                "wave_energy_ratio": te/tp,
                "Tz": np.sqrt(m0/m2),
                "Dp": np.arctan2(b1[a0.argmax()], a1[a0.argmax()]),
                "PeakPSD": a0.max(),
                "te": te,
                "dp_true": np.degrees(dp) % 360,
                "dp_mag": np.degrees(dp+data["declination"]) % 360,
                "a1": a1,
                "b1": b1,
                "a2": (Band["xx"] - Band["yy"]) / denom,
                "b2": -2 * Band["xy"].real / denom,
            }
            print("Calculated Data: ")
            for j in Output["banded"]:
                if np.isscalar(Output["banded"][j]):
                    print(j, "=", Output["banded"][j])
        banded(args.banding)

        print("\n\nCDIP Data: ")
        for j in data["wave"]:
            if np.isscalar(data["wave"][j][i]):
                print(j, "=", data["wave"][j][i])

        ##########################################
        # plotting
        ##########################################

        if(args.raw):
            figure = [
                ["X Acc", "", "m/s^2", data["time"][select], acc["x"]],
                ["Y Acc", "", "m/s^2", data["time"][select], acc["y"]],
                ["Z Acc", "Time (s)", "m/s^2", data["time"][select], acc["z"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.norm and args.graph):
            figure = [
                ["X PSD", "", "", freq_space, PSD["xx"]],
                ["Y PSD", "", "", freq_space, PSD["yy"]],
                ["Z PSD", "freq (Hz)", "", freq_space, PSD["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.banding and args.graph):
            figure = [
                ["X Banded PSD", "", "", freq_midpoints, Band["xx"]],
                ["Y Banded PSD", "", "", freq_midpoints, Band["yy"]],
                ["Z Banded PSD", "freq (Hz)", "", freq_midpoints, Band["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.welch and args.graph):
            figure = [
                ["X Windowed PSD", "", "", wPSD["freq_space"], wPSD["xx"]],
                ["Y Windowed PSD", "", "", wPSD["freq_space"], wPSD["yy"]],
                ["Z Windowed PSD", "freq (Hz)", "",
                 wPSD["freq_space"], wPSD["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.ds):

            if(args.banding):
                A1 = Output["banded"]["a1"]
                B1 = Output["banded"]["b1"]
                A2 = Output["banded"]["a2"]
                B2 = Output["banded"]["b2"]
            if args.banding:
                figure = [
                    ["Directional Spectra", "", "A1",
                        freq_midpoints, [A1, data["wave"]["a1"][i]]],
                    ["", "", "B1", freq_midpoints, [B1, data["wave"]["b1"][i]]],
                    ["", "", "A2", freq_midpoints, [A2, data["wave"]["a2"][i]]],
                    ["", "freq (Hz)", "B2", freq_midpoints,
                     [B2, data["wave"]["b2"][i]]]
                ]
                fig, axs = plt.subplots(nrows=4, ncols=1)
                Plotter(fig, axs, figure)
            else:
                print("Error: please enter calculation option (Welch, Banding, Norm)")

        if(args.welch or args.banding or args.raw or args.ds or args.norm):
            plt.show()

        print("\n--------------------------\n")

        # exit(0)  # comment out if you want to proccess all the blocks of data



def main():
    parser = ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    # type --raw before nc file
    parser.add_argument("--raw", action="store_true", help="Raw acceleration data")
    # type --welch before nc file
    grp.add_argument("--welch", action="store_true", help="Welch Method")
    # type --banding before nc file
    grp.add_argument("--banding", action="store_true", help="Banding")
    # type --DS before nc file
    grp.add_argument("--norm", action="store_true", help="Normal FFT PSD")
    parser.add_argument("--ds", action="store_true",
                        help="Directional Spectrum coefficients")
    parser.add_argument("--graph", action="store_true",
                        help="Turns graphs on")
    parser.add_argument("nc", nargs="+", type=str,
                        help="netCDF file to process")  # typed after commands
    args = parser.parse_args()

    for fn in args.nc:
        process(fn, args)



if __name__ == "__main__":
    main()