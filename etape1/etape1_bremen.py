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

import xarray as xr
import numpy as np

ds = xr.open_dataset(jour)

lats = ds['latitude'].values 
lons = ds['longitude'].values  

new_data_vars = {}

for var in ds.data_vars:
    if var not in ['latitude', 'longitude']:
        # Créer une grille 2D vide remplie de NaN (shape : lat x lon)
        grid = np.full((len(lats), len(lons)), np.nan)

        # Remplir la grille à partir des valeurs
        for i in range(len(lats)):
            lat_idx = np.where(lats == lats[i])[0][0]
            lon_idx = np.where(lons == lons[i])[0][0]

            if grid[lat_idx, lon_idx] == np.nan:
                grid[lat_idx, lon_idx] = ds[var].values[i]

        new_data_vars[var] = (('latitude', 'longitude'), grid)

new_ds = xr.Dataset(
    data_vars=new_data_vars,
    coords={
        'latitude': lats,
        'longitude': lons
    }
)


print(new_ds)