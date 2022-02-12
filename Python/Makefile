# common commands

banding:
# display calculations with banding fft PSD and directional spectral (only one working right now)
	.\cdip_driver.py --banding --ds ".\067.20201225_1200.20201225_1600.nc"
	

welch:	
# display calculations with welch windows fft PSD and directional spectral
	.\cdip_driver.py --welch --hann --ds ".\067.20201225_1200.20201225_1600.nc"

all:
# display with raw acceleration data with normal fft PSD and directional spectral
	.\cdip_driver.py --banding --raw --norm --ds ".\067.20201225_1200.20201225_1600.nc"
	.\cdip_driver.py --welch --hann --raw --norm --ds ".\067.20201225_1200.20201225_1600.nc"