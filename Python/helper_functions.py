#! /usr/bin/env python3
#
#
# URLs:
# https://docs.google.com/document/d/1Uz_xIAVD2M6WeqQQ_x7ycoM3iKENO38S4Bmn6SasHtY/pub
#
# Hs:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/compendium.html
#
# Hs Boxplot:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/annualHs_plot.html
#
# Sea Surface Temperature:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/temperature.html
#
# Polar Spectrum:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/polar.html
#
# Wave Direction and Energy Density by frequency bins:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
#
# XYZ displacements:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html
#
# The Datawell documentation is very useful:
# https://www.datawell.nl/Portals/0/Documents/Manuals/datawell_manual_libdatawell.pdf
#
# Jan 2022
# Alex Pitts, Clayton Surgeon, Benjamin Cha
# In collaboration with Pat Welch

# import argparse
from asyncio.windows_events import NULL
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd



# filename = "./067.20201225_1200.20201225_1600.nc"


def dictMerge(dicts: list) -> dict:
    """takes in a list of dictionaries with matching layouts and keys, then merges the keys into lists for each key """
    final_dict = {}

    Process = []
    Hs = []
    Ta = []
    Tp = []
    wave_energy_ratio = []
    Tz = []
    PeakPSD = []
    te = []
    Dp = []
    dp_mag = []
    A1 = []
    B1 = []
    A2 = []
    B2 = []

    # merge all dictionaries from input list into final_dict{}
    for i in dicts:
        Process.append(i["Process"])
        Hs.append(i["Hs"])
        Ta.append(i["Ta"])
        Tp.append(i["Tp"])
        wave_energy_ratio.append(i["wave_energy_ratio"])
        Tz.append(i["Tz"])
        PeakPSD.append(i["PeakPSD"])
        te.append(i["te"])
        Dp.append(i["Dp"])
        dp_mag.append(i["dp_mag"])
        A1.append(i["A1"])
        B1.append(i["B1"])
        A2.append(i["A2"])
        B2.append(i["B2"])
    
    A1_final = xr.DataArray(
        np.array(A1, dtype=object),
        dims = ("y", "x"),
        coords = {
            "x": (np.arange(1, len(A1[0])+1)).tolist(),
            "y": (np.arange(1, len(A1)+1)).tolist(),
        }
    )
    B1_final = xr.DataArray(
        np.array(B1, dtype=object),
        dims = ("y", "x"),
        coords = {
            "x": (np.arange(1, len(B1[0])+1)).tolist(),
            "y": (np.arange(1, len(B1)+1)).tolist(),
        }
    )
    A2_final = xr.DataArray(
        np.array(A2, dtype=object),
        dims = ("y", "x"),
        coords = {
            "x": (np.arange(1, len(A2[0])+1)).tolist(),
            "y": (np.arange(1, len(A2)+1)).tolist(),
        }
    )
    B2_final = xr.DataArray(
        np.array(B2, dtype=object),
        dims = ("y", "x"),
        coords = {
            "x": (np.arange(1, len(B2[0])+1)).tolist(),
            "y": (np.arange(1, len(B2)+1)).tolist(),
        }
    )
    
    final_dict = {
        "Process": np.array(Process, dtype=object),
        "Hs": np.array(Hs, dtype=object),
        "Ta": np.array(Ta, dtype=object),  # average period
        "Tp": np.array(Tp, dtype=object),  # peak wave period
        "wave_energy_ratio": np.array(wave_energy_ratio, dtype=object),
        "Tz": np.array(Tz, dtype=object),
        # "Dp": np.arctan2(B1[a0.argmax()], A1[a0.argmax()]),
        "PeakPSD": np.array(PeakPSD, dtype=object),
        "te": np.array(te, dtype=object),  # mean energy period
        "Dp": np.array(Dp, dtype=object),
        "dp_mag": np.array(dp_mag, dtype=object),
        "A1": A1_final,
        "B1": B1_final,
        "A2": A2_final,
        "B2": B2_final,
    }
    # print(final_dict)
    # exit(0)
    return final_dict


def Plotter(fig, axs, xy) -> NULL:
    """Takes a Figure from matplotlib, the array for the figure, and a list of data to plot,
    data in xy stored as a list of lists where xy = [["title", "namex", "namey", [x], [y]], [...]] 
    where x and y can be lists of plots themselves
    """
    index = 0
    for i in axs.reshape(-1):
        i.set_title(xy[index][0])
        i.set_xlabel(xy[index][1])
        i.set_ylabel(xy[index][2])
        if isinstance(xy[index][3], list) and isinstance(xy[index][4], list):
            for (j, k) in zip(xy[index][3], xy[index][4]):
                i.plot(j, k)

        elif isinstance(xy[index][3], list):
            for j in xy[index][3]:
                i.plot(j, xy[index][4])

        elif isinstance(xy[index][4], list):
            for j in xy[index][4]:
                i.plot(xy[index][3], j)

        else:
            i.plot(xy[index][3], xy[index][4])


        index += 1
    plt.tight_layout()


def Rolling_mean(x: np.array, w: np.array) -> np.array:
    """Smoothes the raw acceleration data with a rolling mean. 
    Accepts data to be smoothed and a window width for the moving average. 
    """ 
    df = pd.DataFrame({'a': x.tolist()})
    return df['a'].rolling(window=w, center=True).mean().fillna(0).to_numpy()


# new
def Bias(width: int, window: str = "hann") -> np.array:
    """returns a either a boxcar, or hann window"""
    return np.ones(width) if window == "boxcar" else (
        0.5*(1 - np.cos(2*np.pi*np.arange(width) / (width - 0)))
    )


def wfft(data: np.array, width: int, window: str = "hann") -> list[np.array]:
    """Splits the acceleration data into widows, 
    preforms FFTs on them returning a list of all the windows
    """
    bias = Bias(width, window)

    ffts = []
    for i in range(0, data.size-width+1, width//2):
        w = data[i:i+width]
        ffts.append(np.fft.rfft(w*bias))

    return ffts


def wcalcPSD(
        A_FFT_windows: list[np.array],
        B_FFT_windows: list[np.array],
        fs: float,
        window: str) -> np.array:
    """calculates the PSD of the FFT output preformed with the windowing method.
    After calculateing the PSD of each window, the resulting lists are averaged together"""

    width = A_FFT_windows[0].size
    spectrums = np.complex128(np.zeros(width))
    for i in range(len(A_FFT_windows)):
        A = A_FFT_windows[i]
        B = B_FFT_windows[i]

        spectrum = calcPSD(A, B, fs, window=window)
        spectrums += spectrum
    return spectrums / len(A_FFT_windows)

  
def calcPSD(xFFT: np.array, yFFT: np.array, fs: float, window: str) -> np.array:
    "calculates the PSD on an output of a FFT"
    nfft = xFFT.size
    qOdd = nfft % 2
    n = (nfft - qOdd) * 2  # Number of data points input to FFT
    w = Bias(n, window)  # Get the window used
    wSum = (w * w).sum()
    psd = (xFFT.conjugate() * yFFT) / (fs * wSum)
    if not qOdd:       # Even number of FFT bins
        psd[1:] *= 2   # Real FFT -> double for non-zero freq
    else:              # last point unpaired in Nyquist freq
        psd[1:-1] *= 2  # Real FFT -> double for non-zero freq
    return psd


def calcAcceleration(x: np.array, fs: float) -> np.array:
    """converts displacement data to acceleration.
    We need acceleration data because that is
    what we will record from the STM"""
    dx2 = np.zeros(x.shape)
    dx2[2:] = np.diff(np.diff(x))
    dx2[0:2] = dx2[2]
    return dx2 * fs * fs


def Data(args) -> dict:
    """Master data reading function. Reads the .nc file from CDIP.
    The data is stored in dictionary (data), which contains many dictionaries 
    to hold information. Examples include: acceleration data, frequency bounds, 
    expected values calculated by CDIP, etc."""
    FREQ = True
    TXYZ = True
    WAVE = args.compare
    META = True
    TIMEBOUNDS = True

    meta_xr = xr.open_dataset(args.nc[0], group="Meta")  # For water depth
    wave_xr = xr.open_dataset(args.nc[0], group="Wave")
    xyz_xr = xr.open_dataset(args.nc[0], group="XYZ")

    depth = float(meta_xr.WaterDepth)
    declination = float(meta_xr.Declination)
    frequency = float(xyz_xr.SampleRate)

    data = {}
    if META:
        data["frequency"] = frequency
        data["latitude"] = float(meta_xr.DeployLatitude)
        data["longitude"] = float(meta_xr.DeployLongitude)
        data["depth"] = depth
        data["declination"] = declination

    if TXYZ:
        data["time"] = xyz_xr.t.to_numpy()
        data["dis"] = {
            "t": data["time"],
            "x": xyz_xr.x.to_numpy(),
            "y": xyz_xr.y.to_numpy(),
            "z": xyz_xr.z.to_numpy()
        }
       
        data["acc"] = {
            "t": data["time"],
            "x": calcAcceleration(xyz_xr.x.to_numpy(), frequency),
            "y": calcAcceleration(xyz_xr.y.to_numpy(), frequency),
            "z": calcAcceleration(xyz_xr.z.to_numpy(), frequency),
        }

    if WAVE:
        data["wave"] = {
            "sig-height": wave_xr.Hs.to_numpy(),
            "avg-period": wave_xr.Ta.to_numpy(),
            "peak-period": wave_xr.Tp.to_numpy(),
            "mean-zero-upcross-period": wave_xr.Tz.to_numpy(),
            "peak-direction": wave_xr.Dp.to_numpy(),
            "peak-PSD": wave_xr.PeakPSD.to_numpy(),
            "A1": wave_xr.A1.to_numpy(),
            "B1": wave_xr.B1.to_numpy(),
            "A2": wave_xr.A2.to_numpy(),
            "B2": wave_xr.B2.to_numpy(),
        }

    if TIMEBOUNDS:
        # data["wave"]["time-bounds"] = {
        data["time-bounds"] = {
            "lower": wave_xr.TimeBounds[:, 0].to_numpy(),
            "upper": wave_xr.TimeBounds[:, 1].to_numpy()
        }
        data["Timebounds"] = wave_xr.TimeBounds

    if FREQ:
        data["freq"] = {
            "bandwidth": wave_xr.Bandwidth.to_numpy(),
        }
        
        data["freq"]["bounds"] = {
            "lower": wave_xr.FreqBounds[:, 0].to_numpy(),
            "upper": wave_xr.FreqBounds[:, 1].to_numpy(),
            "joint": wave_xr.FreqBounds[:, :].to_numpy()
        }
        data["FreqBounds"] = wave_xr.FreqBounds

    return data