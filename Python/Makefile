# common commands
banding:
# calculations with banding fft PSD 
	.\cdip_driver.py --banding ".\ncFiles\067.20201225_1200.20201225_1600.nc" 

welch:	
# display calculations with welch windows fft PSD and directional spectral
	.\cdip_driver.py --welch --hann ".\ncFiles\067.20201225_1200.20201225_1600.nc" 

output_to_root_dir:	
# display calculations with welch windows fft PSD and directional spectral
	.\cdip_driver.py --welch --hann -o ".\" ".\ncFiles\067.20201225_1200.20201225_1600.nc" 

output_to_folder:	
# display calculations with welch windows fft PSD and directional spectral
	.\cdip_driver.py --welch --hann -o ".\ncFiles" ".\ncFiles\067.20201225_1200.20201225_1600.nc" 

all:
# display with raw acceleration data with normal fft PSD and directional spectral 
	.\cdip_driver.py --welch --hann --raw --norm --ds ".\ncFiles\067.20201225_1200.20201225_1600.nc" 

coverage_testing:
# runs a coverage test on the driver file and helper functions utilizing multiple 
# runs of the program with different calculation methods
	coverage run --omit */site-packages/* .\testing\coverage_test.py 

