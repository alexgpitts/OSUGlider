import os, sys
import matplotlib.pyplot as plt
import numpy as np
path = sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_functions import (Plotter)
import cdip_driver as driver


def test_plotter():
    '''Check for unused feature implemented in our plotter function. We left the functionality
    because someone might find the feature usefull someday. Checks two data sets of X values and
    two data sets of Y values exclusively. ie: X = [[...] [...]] Y = [...] and X = [...] Y = [[...][...]]
    '''
    arr1 = np.arange(20)
    figure = [
        ["X", "label", "label", arr1, arr1],
        ["Y", "label", "label", [arr1, arr1], [arr1, arr1]],
        ["Z", "Time (s)", "label", [arr1, arr1], arr1],
        ["Z", "Time (s)", "label", arr1, [arr1,arr1]]
    ]
    fig, axs = plt.subplots(nrows=4, ncols=1)
    Plotter(fig, axs, figure)

test_plotter()

driver.main(['067.20201225_1200.20201225_1600.nc'])
driver.main(['--welch', '--hann', '--ds', '--graph', '067.20201225_1200.20201225_1600.nc'])
driver.main(['--banding', '--ds', '--raw', '--norm', '067.20201225_1200.20201225_1600.nc'])
driver.main(['--banding', '--ds', '--graph', '--norm', '067.20201225_1200.20201225_1600.nc'])



       
