# Ocean wave measurements (Python Implementation)

This program needs numpy, matplotlib, netCDF4, and xarray installed to run. To display different graphs, change the plotting perimeter boolean values towards the top of the driver file.

This file takes command lines<br /> 
ie: cdip_driver.py [--raw] [--welch | --banding | --norm] [--ds] [--graph] nc [nc ...] <br />
where: <br />
    1) --raw plots the raw acceleration data <br />
    2) --welch, --banding, or --norm chooses the calculation method <br />
    3) --ds plots the directional spectra coefficients compared with CDIPs calculations <br />
    4) --graph chooses if you want to plot the resulting PSD calculated from the chosen calculation method <br />
    5) nc is the .nc file you wish to process <br />

Current Status: <br />
As of now, we have most of the calculations performed on some test data from CDIP. 

Some of our next steps: 
1) Currently we only have the banding option for command line arguments, we need to add welch and normal calculations as well
2) We also need alternative windowing options for banding like we set up for the welch method. 
3) Create a testing suite for running large amounts of CDIP data through the program. 


Example Output:  
![builds](https://github.com/alexgpitts/OSUGlider/blob/main/ProjectImages/python_output.png?raw=true)