# Ocean wave measurements (Python Implementation)

This program needs numpy, matplotlib, netCDF4, and xarray installed to run. To display different graphs, change the plotting perimeter boolean values towards the top of the driver file.

Current Status: 
As of now, we have most of the calculations performed on some test data from CDIP. 

Some of our next steps: 
1) Currently we only have the banding option for command line arguments, we need to add welch and normal calculations as well
2) We also need alternative windowing options for banding like we set up for the welch method. 
3) Create a testing suite for running large amounts of CDIP data through the program. 


Example Output:  
![builds](https://github.com/alexgpitts/OSUGlider/blob/main/ProjectImages/python_output.png?raw=true)