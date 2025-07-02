import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
import cartopy.crs as ccrs

import cartopy.feature as cfeature

DATA_DIR = 'C:/Users/eulal/cours-info/tutorial_TROPOMI_Mines/work_data/official_product'
jour = 13

def distance(lat1d, lon1d, lat2d, lon2d):
    lat1=lat1d*np.pi/180
    lat2=lat2d*np.pi/180
    lon1=lon1d*np.pi/180
    lon2=lon2d*np.pi/180
    return 2*Rterre*np.arcsin(np.sqrt((np.sin((lat2-lat1)/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon2 - lon1)/2))**2))

Rterre = 6378  # km
taillecarre = 250  # km

def selection_carre(jour, temps, lat, lon):

    liste = []

    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    for i in range(len(files)):
        orbital=DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i]
        ds = xr.open_dataset(orbital, group='PRODUCT')
   
        lat_max = min(lat + (taillecarre * 180 / (Rterre * np.pi)), 90)
        lat_min = max(lat - (taillecarre * 180 / (Rterre * np.pi)), -90)

        lat_rad = lat * np.pi / 180
        lon_delta = taillecarre * 180 / (Rterre * np.pi * np.cos(lat_rad))
        lon_max = (lon + lon_delta + 180) % 360 - 180
        lon_min = (lon - lon_delta + 180) % 360 - 180

        mask = ((ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max) &
             (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
           )

    # Vérifier qu'au moins un point est dans le masque
        if mask.any().item():
            carre = ds.where(mask, drop=True)
            liste.append(carre)

    return liste

def selection_carre_optimal(jour, temps, lat, lon):

    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    tmax=0
    i = 0
    while temps > tmax:
        tmax=int(files[i][45:51])
        i += 1
    orbital=DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i+1]
    ds=xr.open_dataset(orbital, group='PRODUCT')

    lat_max = min(lat + (taillecarre * 180 / (Rterre * np.pi)), 90)
    lat_min = max(lat - (taillecarre * 180 / (Rterre * np.pi)), -90)

    lat_rad = lat * np.pi / 180
    lon_delta = taillecarre * 180 / (Rterre * np.pi * np.cos(lat_rad))
    lon_max = (lon + lon_delta + 180) % 360 - 180
    lon_min = (lon - lon_delta + 180) % 360 - 180

    mask = (
        (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max) &
        (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
    )

    # Vérifier qu'au moins un point est dans le masque
    if mask.any().item():
        carre = ds.where(mask, drop=True)
    else:
        carre = None

    return ds, carre, lon_min, lon_max, lat_min, lat_max

def apply_log10(data):
    # let's be careful with zeros, and replace with NaNs
    buf = np.copy(data)
    buf[buf==0] = np.nan
    out = np.log10(buf)
    return out

def tracer_methane(resultat):

    donnees = resultat[1]
    lon_min, lon_max, lat_min, lat_max = resultat[2:]
    img_extent = (lon_min, lon_max, lat_min, lat_max)

    fig = plt.figure(figsize = (7, 7))
    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.set_extent(img_extent, crs = ccrs.PlateCarree())
    ax.coastlines('10m', linewidth=1.5, color='black')
    donnees['log10_of_emissions'] = xr.apply_ufunc(apply_log10,
                                        donnees.methane_mixing_ratio_bias_corrected)
    donnees.log10_of_emissions.plot(x='longitude',
                           y='latitude',
                           antialiased=True,
                           cmap='viridis',
                           transform=ccrs.PlateCarree()
                           )
   
    plt.show()

donnees = selection_carre_optimal(13, 100000, -25.6, 28.67)[1]
print(donnees)
resultat = selection_carre_optimal(13, 100000, -25.6, 28.67)
tracer_methane(resultat)
