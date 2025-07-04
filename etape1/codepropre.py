import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_DIR = r'C:/Users/eulal/cours-info/tutorial_TROPOMI_Mines/work_data/official_product'
jour = 13
Rterre = 6378  # km
taillecarre = 250  # km

def distance(lat1d, lon1d, lat2d, lon2d):
    lat1=lat1d*np.pi/180
    lat2=lat2d*np.pi/180
    lon1=lon1d*np.pi/180
    lon2=lon2d*np.pi/180
    return 2*Rterre*np.arcsin(np.sqrt((np.sin((lat2-lat1)/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon2 - lon1)/2))**2))

def selection_carre(ds, lat, lon):

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

    if mask.any().item():
        carre = ds.where(mask, drop=True)
    else:
        carre = None

    return ds, carre

def selection_cercle(ds, lat, lon):

    mask = (
        distance(lat, lon, ds['latitude'], ds['longitude']) <= 250
    )

    # Vérifier qu'au moins un point est dans le masque
    if mask.any().item():
        carre = ds.where(mask, drop=True)
    else:
        carre = None

    return ds, carre

def recherche_totale(lat, lon):

    liste = []

    for jour in range(13, 21):
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

            if mask.any().item():
                carre = ds.where(mask, drop=True)
                liste.append(carre)
   
    arg_max = liste[0]
    maxi = arg_max['methane_mixing_ratio_bias_corrected'].count().item()
    for i in range(len(liste)):
        non_nan_count = liste[i]['methane_mixing_ratio_bias_corrected'].count().item()
        if non_nan_count>maxi:
            maxi = non_nan_count
            arg_max = liste[i]

    return arg_max

def apply_log10(data):
    # let's be careful with zeros, and replace with NaNs
    buf = np.copy(data)
    buf[buf==0] = np.nan
    out = np.log10(buf)
    return out

def tracer_methane(donnees):

    fig = plt.figure(figsize = (7, 7))
    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    donnees['log10_of_emissions'] = xr.apply_ufunc(apply_log10,
                                        donnees.methane_mixing_ratio_bias_corrected)
    donnees.log10_of_emissions.plot(x='longitude',
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
    plt.title('emission a la latitude'+ str(donnees.attrs['latitude_source']) + "et longitude" + str(donnees.attrs['longitude_source']))
    return(fig)

def final(jour, lat, lon, emission, incertitude):

    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    liste = []
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

        if mask.any().item():
            chemin = orbital

            ds1=xr.open_dataset(chemin, group='PRODUCT')
            ds2=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/DETAILED_RESULTS')
            ds3=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/GEOLOCATIONS')
            ds4=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/INPUT_DATA')

            bon_format = xr.Dataset(
                {
                "methane_mixing_ratio_bias_corrected": (["scanline", "ground_pixel"], ds1['methane_mixing_ratio_bias_corrected'][0].values),
                "longitude_bounds": (['scanline', 'ground_pixel', 'corner'], ds3['longitude_bounds'][0].values),
                "latitude_bounds": (['scanline', 'ground_pixel', 'corner'], ds3['latitude_bounds'][0].values),
                "surface_pressure": (["scanline", "ground_pixel"], ds4['surface_pressure'][0].values),
                "surface_albedo" : (["scanline", "ground_pixel"], ds2['surface_albedo_SWIR'][0].values),
                "eastward_wind": (["scanline", "ground_pixel"], ds4['eastward_wind'][0].values),
                "northward_wind": (["scanline", "ground_pixel"], ds4['northward_wind'][0].values)
                },
            coords={
                "latitude": (['scanline', 'ground_pixel'], ds1['latitude'][0].values),
              "longitude": (['scanline', 'ground_pixel'], ds1['longitude'][0].values),
                "corner" : (['corner'], ds1['corner'].values) ,
                "scanline": (['scanline'], ds1['scanline'].values),
                "ground_pixel": (['ground_pixel'], ds1['ground_pixel'].values)
            },
            attrs={
                "latitude_source": lat,
                "longitude_source": lon,
                "source_rate": emission,
                "incertitude": incertitude
                }
            )
            resultat = selection_carre(bon_format, lat, lon)[1]
            liste.append(resultat)
    return liste

def final_optimal(jour, temps, lat, lon, emission, incertitude):

    files=os.listdir(DATA_DIR + '/' + 'data' + '/' + str(jour))
    tmax = 0
    i = 0
    while temps > tmax:
        tmax = int(files[i][45:51])
        i += 1
    chemin = DATA_DIR + '/' + 'data' + '/' + str(jour) + '/' + files[i-1]

    ds1=xr.open_dataset(chemin, group='PRODUCT')
    ds2=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/DETAILED_RESULTS')
    ds3=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/GEOLOCATIONS')
    ds4=xr.open_dataset(chemin, group='PRODUCT/SUPPORT_DATA/INPUT_DATA')

    bon_format = xr.Dataset(
        {
        "methane_mixing_ratio_bias_corrected": (["scanline", "ground_pixel"], ds1['methane_mixing_ratio_bias_corrected'][0].values),
        "longitude_bounds": (['scanline', 'ground_pixel', 'corner'], ds3['longitude_bounds'][0].values),
        "latitude_bounds": (['scanline', 'ground_pixel', 'corner'], ds3['latitude_bounds'][0].values),
        "surface_pressure": (["scanline", "ground_pixel"], ds4['surface_pressure'][0].values),
        "surface_albedo" : (["scanline", "ground_pixel"], ds2['surface_albedo_SWIR'][0].values),
        "eastward_wind": (["scanline", "ground_pixel"], ds4['eastward_wind'][0].values),
        "northward_wind": (["scanline", "ground_pixel"], ds4['northward_wind'][0].values)
        },
    coords={
        "latitude": (['scanline', 'ground_pixel'], ds1['latitude'][0].values),
        "longitude": (['scanline', 'ground_pixel'], ds1['longitude'][0].values),
        "corner" : (['corner'], ds1['corner'].values) ,
        "scanline": (['scanline'], ds1['scanline'].values),
        "ground_pixel": (['ground_pixel'], ds1['ground_pixel'].values)
    },
    attrs={
        "latitude_source": lat,
        "longitude_source": lon,
        "source_rate": emission,
        "incertitude": incertitude
        }
    )
    resultat = selection_carre(bon_format, lat, lon)[1]
    return(resultat)

#on peut obtenir une sélection cercle en changeant juste la fin de nos fonctions finales


'''
resultat = final(13, -25.6, 28.67, 29, 14)
#resultat = final_optimal(13, -25.6, 28.67, 29, 14)
tracer_methane(resultat)
'''
def panaches():
    df=pd.read_csv(r'C:\Users\alfre\Desktop\Hackaton\TROPOMI_Mines\work_data\SRON_Weekly_Methane_Plumes_2023_wk29_v20230724.csv', sep=',')
    df = df[:3]
    for i in range(len(df)):
        jour = int(str(df['date'][i])[6:])
        lat=df['lat'][i]
        lon=df['lon'][i]
        emission=df['source_rate_t/h'][i]
        uncertainty=df['uncertainty_t/h'][i]
        liste = final(jour, lat, lon, emission, uncertainty)
        
        arg_max = liste[0]
        maxi = arg_max['methane_mixing_ratio_bias_corrected'].count().item()
        for j in range(len(liste)):
            non_nan_count = liste[j]['methane_mixing_ratio_bias_corrected'].count().item()
            if non_nan_count>maxi:
                maxi = non_nan_count
                arg_max = liste[j]

        arg_max.to_netcdf("source"+str(i)+".nc")
        
        fig1 = tracer_methane(arg_max)
        fig1.savefig("source"+str(i)+".jpg", format="jpeg", dpi=300)


panaches()
