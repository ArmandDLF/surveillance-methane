import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

Rterre=6378 #km
taillecarre=250 #km
DATA_DIR = '/Users/alfre/Desktop/Hackaton/TROPOMI_Mines/work_data/official_product'
def selection(jour, temps, lat, lon):
    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    tmax=0
    i = 0
    while temps > tmax:
        tmax=int(files[i][45:51])
        i += 1
    orbital=DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i-1]
    ds=xr.open_dataset(orbital, group='PRODUCT')

    ds.set_index(index=['latitude', 'longitude']).unstack('index')

    lat_max=min(lat+(taillecarre*180/(Rterre*np.pi)), 90)
    lat_min=max(lat-(taillecarre*180/(Rterre*np.pi)), -90)
    lon_max=lon+taillecarre*180/(Rterre*np.pi*np.sin(lat))
    if lon_max > 180:
        lon_max=lon_max%180 - 180
    lon_min=lon-taillecarre*180/(Rterre*np.pi*np.sin(lat))
    if lon_min < -180:
        lon_min=lon_min%180 + 180
    carree=ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    return(ds, carree)

selection(13, 110000, 0.0, 0.0)

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

Rterre=6378 #km
taillecarre=250 #km
DATA_DIR = '/Users/alfre/Desktop/Hackaton/TROPOMI_Mines/work_data/official_product'
def selection(jour, temps, lat, lon):
    files = os.listdir(os.path.join(DATA_DIR, 'data', str(jour)))
    tmax = 0
    i = 0
    while temps > tmax:
        tmax = int(files[i][45:51])
        i += 1
    orbital = os.path.join(DATA_DIR, 'data', str(jour), files[i-1])
    ds = xr.open_dataset(orbital, group='PRODUCT')

    lat_delta = taillecarre * 180 / (Rterre * np.pi)
    lat_max = min(lat + lat_delta, 90)
    lat_min = max(lat - lat_delta, -90)

    if np.abs(lat) < 1e-3:
        lon_delta = 180
    else:
        lon_delta = taillecarre * 180 / (Rterre * np.pi * np.abs(np.sin(np.radians(lat))))

    lon_max = lon + lon_delta
    if lon_max > 180:
        lon_max = (lon_max + 180) % 360 - 180

    lon_min = lon - lon_delta
    if lon_min < -180:
        lon_min = (lon_min + 180) % 360 - 180

    # Sélection avec masque logique
    carree = ds.where(
        (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max) &
        (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max),
        drop=True
    )

    return ds, carree

ds1=selection(13, 110000, 0.0, 0.0)[1]
ds1
def apply_log10(data):
    buf = np.copy(data)
    buf[buf==0] = np.nan
    out = np.log10(buf)
    return out

ds1['log10_of_emissions'] = xr.apply_ufunc(apply_log10, 
                                        ds1.methane_mixing_ratio_bias_corrected)

ds1
fig = plt.figure()
ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ds1.log10_of_emissions.plot(x='lon',
                           y='lat',
                           antialiased=True,
                           cmap='jet',
                           transform=ccrs.PlateCarree(),
                           vmin=-2,
                          vmax=5)


#### les trucs qui fonctionnent

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

def selection_carre(jour, temps, lat, lon):
    Rterre = 6378  # km
    taillecarre = 250  # km

    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    tmax=0
    i = 0
    while temps > tmax:
        tmax=int(files[i][45:51])
        i += 1
    orbital=DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i-1]
    ds=xr.open_dataset(orbital, group='PRODUCT')
    
    lat_max = min(lat + (taillecarre * 180 / (Rterre * np.pi)), 90)
    lat_min = max(lat - (taillecarre * 180 / (Rterre * np.pi)), -90)

    lat_rad = lat * np.pi / 180
    lon_delta = taillecarre * 180 / (Rterre * np.pi * np.cos(lat_rad))
    lon_max = (lon + lon_delta + 180) % 360 - 180
    lon_min = (lon - lon_delta + 180) % 360 - 180

    lat = ds['latitude']
    lon = ds['longitude']

    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )

    # Vérifier qu'au moins un point est dans le masque
    if not mask.any().item():
        print("⚠️ Aucun point trouvé dans la zone sélectionnée.")
        return ds, None

    carre = ds.where(mask, drop=True)

    return ds, carre

selection_carre(13, 10000, 80.0, 0.0)

def distance(lat1d, lon1d, lat2d, lon2d):
    lat1 = lat1d*np.pi/180
    lat2 = lat2d*np.pi/180
    lon1 = lon1d*np.pi/180
    lon2 = lon2d*np.pi/180
    return 2*Rterre*np.arcsin(np.sqrt((np.sin((lat2-lat1)/2))**2 + np.cos(lat1)*np.cas(lat2)*(np.sin((lon2-lon1)/2))**2))


Rterre = 6378  # km
taillecarre = 250  # km

def selection_cercle(jour, temps, lat, lon):


    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    tmax=0
    i = 0
    while temps > tmax:
        tmax=int(files[i][45:51])
        i += 1
    orbital=DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i-1]
    ds=xr.open_dataset(orbital, group='PRODUCT')

    ds['log10_of_emissions'] = xr.apply_ufunc(apply_log10, # first function name
                                        ds.emissions) 

    mask = (
        
    )

    # Vérifier qu'au moins un point est dans le masque
    if not mask.any().item():
        print("⚠️ Aucun point trouvé dans la zone sélectionnée.")
        return ds, None

    cercle = ds.where(mask, drop=True)

    return ds, cercle

selection_cercle(13, 10000, 80.0, 0.0)

