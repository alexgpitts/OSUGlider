# Ocean wave measurements (Python Implementation)

This program needs numpy, matplotlib, netCDF4, and xarray installed to run. To display different graphs, change the plotting perimeter boolean values towards the top of the driver file.

This file takes command lines<br /> 
ie: cdip_driver.py [--welch | --banding] [--hann | --boxcar] [--raw] [--ds] [--graph] nc [nc ...] <br />
where: <br />
    1) --welch, --banding chooses the calculation method <br />
    2) --hann or --boxcar selects the windowing method <br />
    3) --norm shows the normal fft using built in numpy fft method <br />
    4) --raw plots the raw acceleration data <br />
    5) --ds plots the directional spectra coefficients compared with CDIPs calculations <br />
    6) --graph chooses if you want to plot the resulting PSD calculated from the chosen calculation method <br />
    7) nc is the .nc file you wish to process <br />

Current Status: <br />
As of now, we have most of the calculations performed on some test data from CDIP. 

Some of our next steps: 
1) Fix normalization for the welch method using a boxcar window
2) Create a testing suite for running large amounts of CDIP data through the program. 


Example Output:  
![builds](../ProjectImages/python_output.png?raw=true)