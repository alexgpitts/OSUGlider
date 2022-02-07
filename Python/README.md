# Ocean wave measurements (Python Implementation)

This program needs numpy, matplotlib, netCDF4, and xarray installed to run. To display different graphs, change the plotting perimeter boolean values towards the top of the driver file.

Current Status: 
As of now, we have most of the calculations performed on some test data from CDIP. 

Some of our next steps:
1) Apply a smoothing step to the raw acceleration data via a rolling mean algorithm. 
2) Add more options for computing the data including alternative windowing and banding methods. 
3) Add command line arguments for selecting various options to compute data mentioned above. 
4) Create a testing suite for running large amounts of CDIP data through the program. 


Example Output:  
![builds](https://github.com/alexgpitts/OSUGlider/blob/main/ProjectImages/python_output.png?raw=true)