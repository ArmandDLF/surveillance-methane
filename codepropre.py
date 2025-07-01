import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


Rterre = 6378  # km
taillecarre = 250  # km
DATA_DIR = '/Users/alfre/Desktop/Hackaton/TROPOMI_Mines/work_data/official_product'
def selection_carre(jour, lat, lon):
    files = os.listdir(f"{DATA_DIR}/data/{jour}")
    files.sort()
    orbital = f"{DATA_DIR}/data/{jour}/{files[0]}"
    ds = xr.open_dataset(orbital, group='PRODUCT')
    # Calcul des bornes géographiques
    lat_delta = taillecarre * 180 / (Rterre * np.pi)
    lat_max = min(lat + lat_delta, 90)
    lat_min = max(lat - lat_delta, -90)

    lat_rad = lat * np.pi / 180
    lon_delta = taillecarre * 180 / (Rterre * np.pi * np.cos(lat_rad))
    lon_max = (lon + lon_delta + 180) % 360 - 180
    lon_min = (lon - lon_delta + 180) % 360 - 180
    # Masque
    mask = (
        (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max) &
        (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
    )

    if not mask.any().item():
        print("⚠️ Aucun point trouvé dans la zone sélectionnée.")
        return ds, None
    carre = ds.where(mask, drop=True)
    return ds, carre

ds=selection_carre(13, 50.0, 0.0)[0]
carre=selection_carre(13, 50.0, 0.0)[1]
print("Min lat :", carre['latitude'].min().item())
print("Max lat :", carre['latitude'].max().item())
print("Min lat :", ds['latitude'].min().item())
print("Max lat :", ds['latitude'].max().item())
print(ds)


def tracer_methane(carre):
    if 'methane_mixing_ratio_bias_corrected' not in carre:
        print("❌ La variable 'methane_mixing_ratio_bias_corrected' est absente.")
        return
    lat = carre['latitude'].squeeze()  
    lon = carre['longitude'].squeeze()
    methane = carre['methane_mixing_ratio_bias_corrected'].squeeze()
    if 'time' in methane.dims:
        methane = methane.mean(dim='time')
        lat = lat.mean(dim='time')
        lon = lon.mean(dim='time')
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title("Concentration en méthane (bias corrigé)")
    im = ax.pcolormesh(
        lon, lat, methane,
        transform=ccrs.PlateCarree(),
        cmap='viridis'
    )
    plt.colorbar(im, ax=ax, label='ppb')
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    plt.show()

tracer_methane(carre)
