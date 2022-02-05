# import Data from CDIP
from helper_functions import (Data, calcPSD, wcalcPSD, wfft, Plotter)
import numpy as np
import matplotlib.pyplot as plt


# plotting perameters. Change for what you want displayed in graphs
displayPSD = True
displayDS = True
displayRaw = True

# fft perameters
window_type = "hann"


# call master data function to extract all our data from the .nc file
data = Data()


# time bounds holds an array starting and ending times for each analisis block
time_bounds = data["wave"]["time-bounds"]
time = data["time"]

#lists of upper, lower, and midpoint frequencies for banding
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
    acc = {
        "x": data["acc"]["x"][select],      # x is northwards
        "y": data["acc"]["y"][select],      # y is eastwards
        "z": data["acc"]["z"][select]       # z is upwards
    }

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
        # "xz": calcPSD(FFT["x"], FFT["z"], data["frequency"]),
        "zx": calcPSD(FFT["z"], FFT["x"], data["frequency"], "boxcar"),

        # "yz": calcPSD(FFT["y"], FFT["z"], data["frequency"]),
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
    # sig wave height
    ##########################################
    
    a0 = Band["zz"] / np.square(np.square(2 * np.pi * freq_midpoints))

    tp = 1/freq_midpoints[a0.argmax()]
    # a0W = wPSD["zz"][1:65] / np.square(np.square(2 * np.pi * wPSD["freq_space"][1:65]))
    m0 = (a0 * data["freq"]["bandwidth"]).sum()

    # shore side
    mm1 = (a0/freq_midpoints*data["freq"]["bandwidth"]).sum()
    te = mm1/m0 #mean energy period

    wave_energy_ratio = te/tp

    m1 = (a0*freq_midpoints*data["freq"]["bandwidth"]).sum()

    m2 = (a0*np.square(freq_midpoints)*data["freq"]["bandwidth"]).sum()
    ta = m0/m1
    tz = np.sqrt(m0/m2)


    wave = data["wave"]


    # print("Hs from CDIP", float(wave["sig-height"][i]),
    #       "4*sqrt(z.var0)", 4 * np.sqrt(acc["z"].var()),
    #       "4*sqrt(m0)", 4 * np.sqrt(m0))
    print("\nSignificant Wave Height: \n\tExpected value = {0}\n\tCalc using variance = {1},\n\tCalc using m0 = {2}".format(
            float(wave["sig-height"][i]), 4 * np.sqrt(acc["z"].var()), 4 * np.sqrt(m0)
        )
    )

    ##########################################
    # peak psd
    ##########################################
    peakPSD = a0.max()
    print("\nPeakPSD:\n\tFrom CDIP {0}\n\tcalc {1}".format(
            float(wave["peak-PSD"][i]), peakPSD
        )
    )

    ##########################################
    # a1, b1, a2, b2
    ##########################################
    denom = np.sqrt(Band["zz"] * (Band["xx"] + Band["yy"]))

    a1 = Band["zx"].imag / denom
    b1 = -Band["zy"].imag / denom

    denom = Band["xx"] + Band["yy"]

    a2 = (Band["xx"] - Band["yy"]) / denom
    b2 = -2 * Band["xy"].real / denom

    dp = np.arctan2(b1[a0.argmax()], a1[a0.argmax()]) #radians
    
    print("\ndp_true =", np.degrees(dp)%360)
    # print("dp_mag =", np.degrees(dp+data["meta"]["declination"])%360)

    # print(
    #     "a1 = ", a1, "\n expected = ", data["wave"]["a1"], "\n"
    #     "b1 = ", b1, "\n expected = ", data["wave"]["b1"], "\n"
    #     "a2 = ", a2, "\n expected = ", data["wave"]["a2"], "\n"
    #     "b2 = ", b2, "\n expected = ", data["wave"]["b2"], "\n"

    # )

    ##########################################
    # dominant period
    ##########################################
    print("\nDominant Period:\n\tTp from CDIP = {0}\n\tCalc = {1}\n\tCalc not banded {2}".format(
            float(data["wave"]["peak-period"][i]),
            1/freq_midpoints[a0.argmax()],
            1/freq_space[PSD["zz"].argmax()]
        )
    ) 

    ##########################################
    # plotting
    ##########################################

    if(displayRaw):
        figure = [
            ["X Acc", "", "m/s^2", data["time"][select], acc["x"]],
            ["Y Acc", "", "m/s^2", data["time"][select], acc["y"]],
            ["Z Acc", "Time (s)", "m/s^2", data["time"][select], acc["z"]]
        ]
        fig, axs = plt.subplots(nrows=3, ncols=1)
        Plotter(fig, axs, figure)

    if(displayPSD):

        # X
        figure = [
            ["X PSD", "", "", freq_space, PSD["xx"]],
            ["X Banded PSD", "", "", freq_midpoints, Band["xx"]],
            ["X Windowed PSD", "freq (Hz)", "", wPSD["freq_space"], wPSD["xx"]]
        ]
        fig, axs = plt.subplots(nrows=3, ncols=1)
        Plotter(fig, axs, figure)

        # y
        figure = [
            ["Y PSD", "", "", freq_space, PSD["yy"]],
            ["Y Banded PSD", "", "", freq_midpoints, Band["yy"]],
            ["Y Windowed PSD", "freq (Hz)", "", wPSD["freq_space"], wPSD["yy"]]
        ]
        fig, axs = plt.subplots(nrows=3, ncols=1)
        Plotter(fig, axs, figure)

        # Z
        figure = [
            ["Z PSD", "", "", freq_space, PSD["zz"]],
            ["Z Banded PSD", "", "", freq_midpoints, Band["zz"]],
            ["Z Windowed PSD", "freq (Hz)", "", wPSD["freq_space"], wPSD["zz"]]
        ]
        fig, axs = plt.subplots(nrows=3, ncols=1)
        Plotter(fig, axs, figure)


    if(displayDS):
        figure = [
            ["Directional Spectra", "", "A1", freq_midpoints, [a1, data["wave"]["a1"][i]]],
            ["", "", "B1", freq_midpoints, [b1, data["wave"]["b1"][i]] ],
            ["", "", "A2", freq_midpoints, [a2, data["wave"]["a2"][i]]],
            ["", "freq (Hz)", "B2", freq_midpoints, [b2, data["wave"]["b2"][i]]]
        ]
        fig, axs = plt.subplots(nrows=4, ncols=1)
        Plotter(fig, axs, figure)


    if(displayDS or displayPSD or displayRaw):
        plt.show()

    print("\n--------------------------\n")

    exit(0) #comment out if you want to proccess all the blocks of data