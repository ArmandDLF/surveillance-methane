import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_DIR = 'C:/Users/eulal/cours-info/tutorial_TROPOMI_Mines/work_data/bremen_product/data' 
jour = DATA_DIR + '/' + 'ESACCI-GHG-L2-CH4_CO-TROPOMI-WFMD-20230713-fv4.nc'
ds = xr.open_dataset(jour)
print(ds)
print(ds.data_vars)
