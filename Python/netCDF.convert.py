# converts CDIP .nc file from displacement to acceleration. Function exists in "helper_functions.py"
# creates a new netCDF file that contains all the same data as before, just with acc inside the xyz group instead of displacement
# command line args would be an input and output file name. 

# python .\netCDF.convert.py --group="Meta" --output="test.nc" ".\ncFiles\067.20201225_1200.20201225_1600.nc"

# python .\netCDF.convert.py --group=Meta --group=Wave --group=XYZ --ouput=test.nc ".\ncFiles\067.20201225_1200.20201225_1600.nc"

from ast import arg
import re
from helper_functions import (
    Data, Rolling_mean, calcPSD, wcalcPSD, wfft, Plotter, Bias)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import xarray as xr
import netCDF4 as nc4
import os
import csv
import time

# Overview:
# Finds a CSV file in current directory, parses it and stores it into an nc file
# It also allows the user to edit and overwrite groups using arguments. 

"""
Displacement to Acceleration
"""
def calcAcceleration(x: np.array, fs: float) -> np.array:
    """converts displacement data to acceleration.
    We need acceleration data because that is
    what we will record from the STM"""
    dx2 = np.zeros(x.shape)
    dx2[2:] = np.diff(np.diff(x))
    dx2[0:2] = dx2[2]
    return dx2 * fs * fs

"""
Finds a CSV file in current directory, parses it and stores it in nc file
"""
def parse_csv(file_name: str, output: str, group: str) -> None:
    if (not os.path.isfile(file_name)): # if nc file isn't in current directory 
        raise Exception(f"Input CSV file, {file_name}, not found!") # throw error if file isn't found
    
    # global attribute
    ds = xr.Dataset(
        attrs=dict(
            date_created=str(np.datetime64(round(1e3*time.time()), "ms")),
            ),
        )

    # os.listdir
    # return a list of files in directory thats given
    # glob - look up the module for csv files
    # return all csv files

    if (not os.path.isfile(output)): # if this file is not found in the directory
        ds.to_netcdf(output, mode="w") # create the file

    # reads csv file, stores it in a dataframe then into a nc file
    data = pd.read_csv(file_name)
    ds = xr.Dataset.from_dataframe(data) # format and stores csv file into dataset
    ds.to_netcdf(output, mode="a", group=group)


"""
Finds CSV file to parse
"""
def store_data(args: ArgumentParser) -> None:
    # # if group is entered and csv file is found in the current directory, then parse that file
    if ("Meta" in args.group):
        parse_csv(args.meta, args.output, "Meta") # be a place to add attributes by using dict
    if ("Wave" in args.group):
        parse_csv(args.wave, args.output, "Wave")
    if ("XYZ" in args.group):
        parse_csv(args.xyz, args.output, "XYZ")

"""
Stores acceleration data in nc file
"""
def acceleration_XYZ(fn: str, args: ArgumentParser) -> None:
    if ("XYZ" in args.group): # prevents a clash if acc XYZ is already written in
        return

    xyz_xr = xr.open_dataset(args.nc[0], group="XYZ")
    frequency = float(xyz_xr.SampleRate)

    data = Data(args.nc[0])
    # # calculates for acceleration and stores in XYZ
    ds = xr.Dataset(
        data_vars=dict(
            x=(("time",), calcAcceleration(data["dis"]["x"], frequency), {"units": "m/s^2"}),
            y=(("time",), calcAcceleration(data["dis"]["y"], frequency), {"units": "m/s^2"}),
            z=(("time",), calcAcceleration(data["dis"]["z"], frequency), {"units": "m/s^2"}),
            ),
        attrs=dict(
            comment="Acceleration Values",
            ),
        )
    ds.to_netcdf(args.output, mode="a", group="XYZ") # writes acceleration data to nc file
    # print("ONE: ", calcAcceleration(data["dis"]["x"], frequency))
    # print("TWO: ", data["acc"]["x"])
    

"""
User can overwrite more than one group using these arguments
python .\netCDF.convert.py --group=Meta --group=Wave --group=XYZ --output=test3.nc ".\ncFiles\067.20201225_1200.20201225_1600.nc" 
"""

def main(raw_args=None):   
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True,
                    help="netCDF file write out too")  # typed after commands
    parser.add_argument("nc", nargs=1, type=str,
                       help="netCDF file to process")  # typed after commands
    parser.add_argument("--meta", type=str, metavar="meta.csv", default="meta.csv", help="For Meta Group")
    parser.add_argument("--wave", type=str, metavar="wave.csv", default="wave.csv", help="For Wave Group")
    parser.add_argument("--xyz", type=str, metavar="acceleration.csv", default="acceleration.csv", help="For XYZ Group")
    parser.add_argument("--group", type=str, action="append", required=True, choices=("Meta", "Wave", "XYZ"), help="Enter Meta, Wave or XYZ")
    args = parser.parse_args(raw_args)


    for fn in args.nc:
        acceleration_XYZ(fn, args) 
        store_data(args) 

if __name__ == "__main__":
    main()
