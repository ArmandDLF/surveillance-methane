import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import cartopy.feature as cfeature
import codepropre

Rterre=6371 #km
taillecarre=250 #km
DATA_DIRB =r'C:/Users/eulal/cours-info/tutorial_TROPOMI_Mines/work_data/bremen_product/data/ESACCI-GHG-L2-CH4_CO-TROPOMI-WFMD-20230713-fv4.nc'

ds=xr.open_dataset(DATA_DIRB)
Rterre = 6378  # km
taillecarre = 250  # km

donnees= codepropre.selection_carre(ds, -30, 20)[1]

lats = donnees['latitude'].values
lons = donnees['longitude'].values  

new_data_vars = {}

for var in donnees.data_vars:
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


def tracer_methane_bremen(new_ds):
    new_ds_sorted = new_ds.sortby(['latitude', 'longitude'])
    fig = plt.figure(figsize = (7, 7))
    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    new_ds_sorted.xch4.plot(x='longitude',
                           y='latitude',
                           antialiased=True,
                           cmap='jet',
                           transform=ccrs.PlateCarree(),
                           vmin = 3.25,
                           vmax = 3.30
                           )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white')
    plt.show()
    return(fig)

tracer_methane_bremen(new_ds)
